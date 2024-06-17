from __future__ import annotations

import pickle
import random
from pathlib import Path
import gc

import seaborn as sns
import click
import numpy as np
import torch
from neptune.types import File
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM, DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
import neptune

from adaptive_pruning.pruning import (
    prune_attention_heads,
    prune_attention_layers,
    prune_ffn_neurons,
    prune_ffn_layers,
    prune_hidden_state,
    select_to_prune_attention_heads,
    select_to_prune_attention_layers,
    select_to_prune_ffn_neurons,
    select_to_prune_ffn_layers, select_to_prune_hidden_states,
)
from adaptive_pruning.importance import (
    ComponentsImportance,
    ComponentsInfo,
    collect_random_numbers,
    collect_activations,
    collect_weight_magnitudes,
    collect_mask_gradients,
    info_to_mean,
    info_to_max,
    info_to_fisher,
    info_to_entropy, info_to_minus_entropy,
)
from adaptive_pruning.utils import count_parameters, format_number, count_flops_macs_params, tensor_to_list
from utils import get_bookcorpus, set_random_seed, evaluate_model

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE and not torch.backends.cuda.matmul.allow_tf32
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.backends.cuda.matmul.allow_tf32
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")
print(f"FP16_AVAILABLE: {IS_FP16_AVAILABLE}")
print(f"BF16_AVAILABLE: {IS_BF16_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@click.command()
@click.option("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
@click.option("--pruning_dataset", type=str, default="bookcorpus")
@click.option("--batch_size", type=int, default=32)
@click.option("--pruning_components", type=str, default="attention_heads")  # attention_heads, attention_layers, ffn_neurons, ffn_layers, hidden_state, all or splited by +
@click.option("--how_to_collect", type=str, default="grads")  # grads or activations or random
@click.option("--how_to_average", type=str, default="fisher_info")  # fisher_info or mean or entropy
@click.option("--how_to_overlap", type=str, default="fixed")  # fixed, relative, meta
@click.option("--pruning_ratio", type=float, default=0.5)
@click.option("--num_samples", type=int, default=256)
@click.option("--use_cache", is_flag=True, default=False)
@click.option("--seed", type=int, default=0)
@click.option("--evaluate", is_flag=True, default=False)
def main(
    base_model: str,
    pruning_dataset: str,
    batch_size: int,
    how_to_collect: str,
    how_to_average: str,
    how_to_overlap: str,
    pruning_components: str,
    pruning_ratio: float,
    num_samples: int,
    use_cache: bool,
    seed: int,
    evaluate: bool,
) -> None:
    set_random_seed(seed)

    if pruning_components == "all":
        pruning_components = "attention_heads+attention_layers+ffn_neurons+ffn_layers+hidden_state"
    pruning_components = pruning_components.split("+")
    do_prune_attention_heads = "attention_heads" in pruning_components
    do_prune_attention_heads_uniform = "attention_heads_uniform" in pruning_components
    do_prune_attention_layers = "attention_layers" in pruning_components
    do_prune_ffn_neurons = "ffn_neurons" in pruning_components
    do_prune_ffn_neurons_uniform = "ffn_neurons_uniform" in pruning_components
    do_prune_ffn_layers = "ffn_layers" in pruning_components
    do_prune_hidden_state = "hidden_state" in pruning_components

    assert (
        do_prune_attention_heads
        or do_prune_attention_heads_uniform
        or do_prune_attention_layers
        or do_prune_ffn_neurons
        or do_prune_ffn_neurons_uniform
        or do_prune_ffn_layers
        or do_prune_hidden_state
    ), "Need to select at least one pruning method"
    assert not (do_prune_attention_heads and do_prune_attention_heads_uniform), "Can't do both head and uniform head"
    assert not (do_prune_ffn_neurons and do_prune_ffn_neurons_uniform), "Can't do both neuron and uniform neuron"

    # setup logging
    neptune_run = neptune.init_run(tags=['our', base_model, how_to_collect, how_to_average, how_to_overlap])
    method = [
        "attn-heads" if do_prune_attention_heads else None,
        "attn-heads-uniform" if do_prune_attention_heads_uniform else None,
        "attn-layers" if do_prune_attention_layers else None,
        "ffn-neurons" if do_prune_ffn_neurons else None,
        "ffn-neurons-uniform" if do_prune_ffn_neurons_uniform else None,
        "ffn-layers" if do_prune_ffn_layers else None,
        "hidden-state" if do_prune_hidden_state else None,
        how_to_average,
        how_to_overlap,
    ]
    method = "+".join([m for m in method if m ])
    neptune_run["parameters"] = {
        "base_model": base_model,
        "lib": "our",
        "method": method,
        "ratio": pruning_ratio,
        "pruning_dataset": pruning_dataset,
        "batch_size": batch_size,
        "pruning_components": pruning_components,
        "how_to_collect": how_to_collect,
        "how_to_average": how_to_average,
        "how_to_overlap": how_to_overlap,
        "num_samples": num_samples,
    }

    # Load the finetuned model and the corresponding tokenizer
    print(f"Loading model {base_model}...")
    config = AutoConfig.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, config=config)
    if IS_CUDA_AVAILABLE:
        model = model.to(device="cuda", non_blocking=True, dtype=torch.float16)
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    print(f"Number of parameters: {format_number(count_parameters(model))}")
    flops, macs, params = count_flops_macs_params(model, tokenizer)
    print(f"FLOPs: {format_number(flops)}, MACs: {format_number(macs)}, Params: {format_number(params)} ({params})")
    print("Model loaded")

    # load dataset
    print(f"Loading dataset {pruning_dataset}...")
    if pruning_dataset == "bookcorpus":
        dataset = get_bookcorpus(tokenizer, num_samples, 128)
    else:
        raise ValueError(f"Unknown pruning_dataset: {pruning_dataset}")
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print(dataset)

    sample_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"Sample dataloader: {len(dataset)} samples")
    print("Dataset loaded")

    # print model with sample input
    # summary(model, input_size=(batch_size, 512), dtypes=['torch.IntTensor'], depth=7, device=model.device)
    print(model)

    print("-" * 80)

    components_info = None
    dataset_model_collect_hash = "info_" + "dataset" + str(dataset._fingerprint) + "_" + base_model.replace("/", "__") + "_" + how_to_collect
    if use_cache and Path(f"results/{dataset_model_collect_hash}.pickle").exists():
        print(f"Loading cached {how_to_collect} from {dataset_model_collect_hash}.pickle...")
        components_info = pickle.load(open(f"results/{dataset_model_collect_hash}.pickle", "rb"))

    if components_info is None:
        if how_to_collect == "grads":
            components_info = collect_mask_gradients(model, sample_dataloader)
        elif how_to_collect == "full_grads":
            components_info = collect_full_gradients(model, sample_dataloader)
        elif how_to_collect == "activations":
            components_info = collect_activations(model, sample_dataloader)
        elif how_to_collect == "weights":
            components_info = collect_weight_magnitudes(model)
        elif how_to_collect == "random":
            components_info = collect_random_numbers(model)
        else:
            assert False, f"Unknown how_to_collect: {how_to_collect}"

        print(f"Saving {how_to_collect} to {dataset_model_collect_hash}.pickle...")
        pickle.dump(components_info, open(f"results/{dataset_model_collect_hash}.pickle", "wb"))

    if how_to_average == "fisher_info":
        components_importance = info_to_fisher(components_info)
    elif how_to_average == "mean":
        components_importance = info_to_mean(components_info)
    elif how_to_average == "max":
        components_importance = info_to_max(components_info)
    elif how_to_average == "entropy":
        components_importance = info_to_entropy(components_info)
    elif how_to_average == "minus_entropy":
        components_importance = info_to_minus_entropy(components_info)
    else:
        assert False, f"Unknown how_to_average: {how_to_average}"
    relative_meta_importance = torch.nn.functional.softmax(components_importance.meta_importance.cpu().to(torch.float32), dim=0)

    # Log importance
    for name, info, importance in [
        ("attention_heads", components_info.attention_heads_info, components_importance.attention_heads_importance),
        ("attention_layers", components_info.attention_layers_info, components_importance.attention_layers_importance),
        ("ffn_neurons", components_info.ffn_neurons_info, components_importance.ffn_neurons_importance),
        ("ffn_layers", components_info.ffn_layers_info, components_importance.ffn_layers_importance),
        ("hidden_states", components_info.hidden_states_info, components_importance.hidden_states_importance),
        ("meta", components_info.meta_info, components_importance.meta_importance),
    ]:
        neptune_run[f"info/type"] = how_to_collect + "+" + how_to_average + "+" + how_to_overlap
        neptune_run[f"info/{name}_importance_pickle"].upload(File.as_pickle(tensor_to_list(importance.cpu())))
        neptune_run[f"info/{name}_info_pickle"].upload(File.as_pickle(tensor_to_list(info.cpu())))

        numpy_image_importance = importance.cpu().numpy()
        numpy_image_importance = numpy_image_importance if len(numpy_image_importance.shape) > 1 else numpy_image_importance[None, :]
        neptune_run[f"info/{name}_importance_heatmap"].upload(File.as_image(sns.heatmap(numpy_image_importance, annot=False).get_figure()))

    # Print average importance
    print(f"Average importance for {how_to_collect}/{how_to_average}/{how_to_overlap}:")
    for layer in range(config.num_hidden_layers):
        print(f"> Layer {layer}:")
        print(f"  Attention Head:            {components_importance.attention_heads_importance[layer].mean().item()}")
        print(f"  Attention Layer sum Heads: {components_importance.attention_heads_importance[layer].sum().item()}")
        print(f"  Attention Layer:           {components_importance.attention_layers_importance[layer].mean().item()}")
        print(f"  FF Neuron:                 {components_importance.ffn_neurons_importance[layer].mean().item()}")
        print(f"  FF Layer sum Neurons:      {components_importance.ffn_neurons_importance[layer].sum().item()}")
        print(f"  FF Layer:                  {components_importance.ffn_layers_importance[layer].mean().item()}")
    print(f"> Hidden State: {components_importance.hidden_states_importance.mean().item()}")
    print(f"> Meta:")
    for meta_name, meta_value in [
        ("attn-head", components_importance.meta_importance[0]),
        ("attn-layer", components_importance.meta_importance[1]),
        ("ffn-neuron", components_importance.meta_importance[2]),
        ("ffn-layer", components_importance.meta_importance[3]),
        ("hidden-state", components_importance.meta_importance[4]),
    ]:
        print(f"  {meta_name}:\t {meta_value}")

    # Select pruning percent for each component type
    print(f"Select pruning percent for each component type, with {how_to_overlap} overlap")
    print(f"Base pruning ratio: {pruning_ratio}")
    if how_to_overlap == "fixed":
        # Prune same rate of specified components (e.g. 20% of attention heads, 20% of ffn neurons)
        attention_heads_pruning_ratio = pruning_ratio if do_prune_attention_heads else 0
        attention_layers_pruning_ratio = pruning_ratio if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = pruning_ratio if do_prune_ffn_neurons else 0
        ffn_layers_pruning_ratio = pruning_ratio if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio if do_prune_hidden_state else 0
    elif how_to_overlap == "fixed_x2_x05":
        # Multiply attention heads deletion, decrease ffn neurons deletion
        attention_heads_pruning_ratio = min(pruning_ratio * 2, 0.9) if do_prune_attention_heads else 0
        attention_layers_pruning_ratio = pruning_ratio if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = min(pruning_ratio * 0.5, 0.9) if do_prune_ffn_neurons else 0
        ffn_layers_pruning_ratio = pruning_ratio if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio if do_prune_hidden_state else 0
    elif how_to_overlap == "fixed_x05_x2":
        # Multiply ffn neurons deletion, decrease attention heads deletion
        attention_heads_pruning_ratio = min(pruning_ratio * 0.5, 0.9) if do_prune_attention_heads else 0
        attention_layers_pruning_ratio = pruning_ratio if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = min(pruning_ratio * 2, 0.9) if do_prune_ffn_neurons else 0
        ffn_layers_pruning_ratio = pruning_ratio if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio if do_prune_hidden_state else 0
    elif how_to_overlap == "random":
        # multiple random pruning ratios on random [0, 2] * pruning_ratio
        attention_heads_pruning_ratio = min(pruning_ratio * random.random() * 2, 0.9) if do_prune_attention_heads else 0
        attention_layers_pruning_ratio = min(pruning_ratio * random.random() * 2, 0.9) if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = min(pruning_ratio * random.random() * 2, 0.9) if do_prune_ffn_neurons else 0
        ffn_layers_pruning_ratio = min(pruning_ratio * random.random() * 2, 0.9) if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = min(pruning_ratio * random.random() * 2, 0.9) if do_prune_hidden_state else 0
    elif how_to_overlap == "relative":
        # Use actual relative importance to calculate pruning ratio
        relative_importance = torch.tensor([
            components_importance.attention_heads_importance.cpu().mean().item(),
            components_importance.attention_layers_importance.cpu().mean().item(),
            components_importance.ffn_neurons_importance.cpu().mean().item(),
            components_importance.ffn_layers_importance.cpu().mean().item(),
            components_importance.hidden_states_importance.cpu().mean().item(),
        ])
        relative_pruning_coefficient = torch.softmax(-1 * relative_importance, dim=0) * relative_importance.shape[0]
        print('Relative importance', relative_importance)
        print('Relative pruning coefficient', relative_pruning_coefficient)
        attention_heads_pruning_ratio = pruning_ratio * relative_pruning_coefficient[0].item() if do_prune_attention_heads else 0
        attention_layers_pruning_ratio = pruning_ratio * relative_pruning_coefficient[1].item() if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = pruning_ratio * relative_pruning_coefficient[2].item() if do_prune_ffn_neurons else 0
        ffn_layers_pruning_ratio = pruning_ratio * relative_pruning_coefficient[3].item() if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio * relative_pruning_coefficient[4].item() if do_prune_hidden_state else 0
    elif how_to_overlap == "relative_per_param":
        # Use actual relative importance to calculate pruning ratio
        # Use number of possible pruning parameters to calculate pruning ratio
        # adjust for number of params in the component
        _num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
        _single_head_size = config.hidden_size // config.num_attention_heads
        # k + v of the same size and q and output of the same size
        num_parameters_attention_head_group = 2 * _single_head_size * config.hidden_size + 2 * _single_head_size * _num_q_per_kv * config.hidden_size
        num_parameters_attention_layer = num_parameters_attention_head_group * config.num_key_value_heads # + bias
        # gate, input, output
        num_parameters_ffn_neuron = 3 * config.hidden_size
        num_parameters_ffn_layer = num_parameters_ffn_neuron * config.intermediate_size # + bias
        # hidden state = emb + 2 norms, attention, ffn
        num_parameters_hidden_state = config.vocab_size + config.num_hidden_layers * (2 + 2*_single_head_size*config.num_key_value_heads + 2*_single_head_size*config.num_attention_heads + 3*config.intermediate_size)
        total_parameters = num_parameters_hidden_state * config.hidden_size + config.num_hidden_layers * (num_parameters_attention_layer + num_parameters_ffn_layer)

        relative_importance = torch.tensor([
            components_importance.attention_heads_importance.cpu().mean().item() / num_parameters_attention_head_group,
            components_importance.attention_layers_importance.cpu().mean().item() / num_parameters_attention_layer,
            components_importance.ffn_neurons_importance.cpu().mean().item() / num_parameters_ffn_neuron,
            components_importance.ffn_layers_importance.cpu().mean().item() / num_parameters_ffn_layer,
            components_importance.hidden_states_importance.cpu().mean().item() / num_parameters_hidden_state,
        ])
        relative_pruning_coefficient = torch.softmax(-1 * relative_importance, dim=0) * relative_importance.shape[0]
        print('Relative importance', relative_importance)
        print('Relative pruning coefficient', relative_pruning_coefficient)

        attention_heads_pruning_ratio = pruning_ratio * relative_pruning_coefficient[0].item() if do_prune_attention_heads else 0
        attention_layers_pruning_ratio = pruning_ratio * relative_pruning_coefficient[1].item() if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = pruning_ratio * relative_pruning_coefficient[2].item() if do_prune_ffn_neurons else 0
        ffn_layers_pruning_ratio = pruning_ratio * relative_pruning_coefficient[3].item() if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio * relative_pruning_coefficient[4].item() if do_prune_hidden_state else 0
    elif how_to_overlap == "meta":
        # change the pruning ratio based on the relative importance of the components
        # e.g. given pruning ration of 20% and attention importance 2 times lower than ffn importance - prune 30% of attention and 10% of ffn
        relative_meta_pruning_coefficient = torch.softmax(-1 * relative_meta_importance, dim=0) * relative_meta_importance.shape[0]
        print('Relative meta importance', relative_meta_importance)
        print('Relative meta pruning coefficient', relative_meta_pruning_coefficient)
        attention_heads_pruning_ratio = pruning_ratio * relative_meta_pruning_coefficient[0].item() if do_prune_attention_heads else 0
        attention_layers_pruning_ratio = pruning_ratio * relative_meta_pruning_coefficient[1].item() if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = pruning_ratio * relative_meta_pruning_coefficient[2].item() if do_prune_ffn_neurons else 0
        ffn_layers_pruning_ratio = pruning_ratio * relative_meta_pruning_coefficient[3].item() if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio * relative_meta_pruning_coefficient[4].item() if do_prune_hidden_state else 0
    else:
        assert False, f"Unknown how_to_overlap: {how_to_overlap}"
    print('  attention_heads_pruning_ratio', attention_heads_pruning_ratio)
    print('  attention_layers_pruning_ratio', attention_layers_pruning_ratio)
    print('  ffn_neurons_pruning_ratio', ffn_neurons_pruning_ratio)
    print('  ffn_layers_pruning_ratio', ffn_layers_pruning_ratio)
    print('  hidden_states_pruning_ratio', hidden_states_pruning_ratio)


    # Select components to prune
    attention_layers_to_prune = select_to_prune_attention_layers(
        components_importance.attention_layers_importance, attention_layers_pruning_ratio,
    )
    attention_heads_to_prune = select_to_prune_attention_heads(
        components_importance.attention_heads_importance,
        attention_heads_pruning_ratio,
        uniform_among_layers=do_prune_attention_heads_uniform,
        key_value_group_size=config.num_attention_heads // config.num_key_value_heads,
    )
    attention_heads_to_prune = {
        layer: heads
        for layer, heads in attention_heads_to_prune.items()
        if layer not in attention_layers_to_prune
    }

    ffn_layers_to_prune = select_to_prune_ffn_layers(
        components_importance.ffn_layers_importance, ffn_layers_pruning_ratio,
    )
    neurons_to_prune = select_to_prune_ffn_neurons(
        components_importance.ffn_neurons_importance,
        ffn_neurons_pruning_ratio,
        uniform_among_layers=do_prune_ffn_neurons_uniform,
    )
    neurons_to_prune = {
        layer: neurons for layer, neurons in neurons_to_prune.items() if layer not in attention_layers_to_prune
    }
    hidden_states_to_prune = select_to_prune_hidden_states(
        components_importance.hidden_states_importance, hidden_states_pruning_ratio,
    )

    # prune
    print("-" * 80)
    print(
        f"Pruning {(pruning_ratio) * 100:.0f}% with {(1-pruning_ratio) * 100:.0f}% remain "
        f"on {how_to_collect}/{how_to_average}/{how_to_overlap}"
    )

    if do_prune_attention_layers:
        # actually prune layers
        prune_attention_layers(model.model, attention_layers_to_prune)
        # print layers deleted
        total_percent_layers_deleted = len(attention_layers_to_prune) / model.config.num_hidden_layers
        print(
            f"Total: {total_percent_layers_deleted * 100:.2f}% attention layers deleted, "
            f"{(1 - total_percent_layers_deleted) * 100:.2f}% remain"
        )

    if do_prune_attention_heads or do_prune_attention_heads_uniform:
        # actually prune heads
        prune_attention_heads(model.model, attention_heads_to_prune)
        # print layers and number of heads deleted
        total_num_heads_deleted = 0
        for layer in range(config.num_hidden_layers):
            head_dim = model.config.hidden_size // model.config.num_attention_heads
            num_heads_deleted = (
                config.num_attention_heads - model.model.layers[layer].self_attn.o_proj.in_features // head_dim
            )
            num_grouped_heads_deleted = (
                config.num_key_value_heads - model.model.layers[layer].self_attn.k_proj.out_features // head_dim
            )
            total_num_heads_deleted += num_heads_deleted
            print(f"> Layer {layer}: {num_heads_deleted} attention heads deleted ({num_grouped_heads_deleted} grouped heads deleted)")
        total_percent_heads_deleted = total_num_heads_deleted / (
            config.num_attention_heads * config.num_hidden_layers
        )
        print(
            f"Total: {total_percent_heads_deleted * 100:.2f}% attention heads deleted, "
            f"{(1 - total_percent_heads_deleted) * 100:.2f}% remain"
        )

    if do_prune_ffn_layers:
        # actually prune layers
        prune_ffn_layers(model.model, ffn_layers_to_prune)
        # print layers deleted
        total_percent_layers_deleted = len(ffn_layers_to_prune) / model.config.num_hidden_layers
        print(
            f"Total: {total_percent_layers_deleted * 100:.2f}% ffn layers deleted, "
            f"{(1 - total_percent_layers_deleted) * 100:.2f}% remain"
        )

    if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform:
        # actually prune neurons
        prune_ffn_neurons(model.model, neurons_to_prune)
        # print layers and number of neurons deleted in FF
        total_num_neurons_deleted = 0
        for layer in range(config.num_hidden_layers):
            num_neurons_deleted = (
                config.intermediate_size - model.model.layers[layer].mlp.up_proj.out_features
            )
            total_num_neurons_deleted += num_neurons_deleted
            print(f"> Layer {layer}: {num_neurons_deleted} neurons deleted")
        total_percent_neurons_deleted = total_num_neurons_deleted / (
            config.intermediate_size * config.num_hidden_layers
        )
        print(
            f"Total: {total_percent_neurons_deleted * 100:.2f}% neurons deleted, "
            f"{(1 - total_percent_neurons_deleted) * 100:.2f}% remain"
        )

    gc.collect()

    # Log pruned model
    if evaluate:
        print("\n==================Evaluation after Pruning==================\n")
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            inference_dataset='bookcorpus',
            name=f"{base_model}-pruning-{pruning_ratio}-{how_to_collect}-{how_to_average}",
            device='cuda' if IS_CUDA_AVAILABLE else 'cpu',
        )
        neptune_run["evaluation"] = eval_results


    neptune_run.stop()


if __name__ == "__main__":
    if IS_FP16_AVAILABLE:
        with torch.cuda.amp.autocast():
            main()
    else:
        main()
