from __future__ import annotations

from typing import Optional

import torch
import typer
from neptune.types import File
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling

from adaptive_pruning.importance import (
    collect_activations,
    collect_mask_gradients,
    collect_random_numbers,
    collect_weight_magnitudes,
    info_to_entropy,
    info_to_fisher,
    info_to_max,
    info_to_mean,
    info_to_minus_entropy,
)
from adaptive_pruning.pruning import (
    prune_attention_heads,
    prune_attention_layers,
    prune_ffn_layers,
    prune_ffn_neurons,
    select_to_prune_attention_heads,
    select_to_prune_attention_layers,
    select_to_prune_ffn_layers,
    select_to_prune_ffn_neurons,
    select_to_prune_hidden_states,
)
from adaptive_pruning.utils import (
    count_flops_macs_params,
    measure_model_stats,
    print_components_info_importance,
    tensor_to_list,
)
from utils import (
    create_neptune_run,
    evaluate_model,
    fix_neptune_overflow_recursively,
    get_tokenized_dataset,
    save_model_tokenizer,
    set_random_seed,
)


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(
    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    pruning_dataset: str = "bookcorpus",
    batch_size: int = 32,
    how_to_collect: str = "grads",  # grads or activations or random
    how_to_average: str = "fisher_info",  # fisher_info, sum, mean, max or entropy
    how_to_overlap: str = "fixed",  # fixed, relative, meta
    pruning_components: str = "attn_heads",  # attn_heads, attn_layers, ffn_neurons, ffn_layers, hidden_state, all
    pruning_ratio: float = 0.5,
    num_samples: int = 256,
    seed: int = 42,
    evaluate_on: Optional[str] = "perplexity+full+bias",
    save_model_as: Optional[str] = None,
) -> None:
    set_random_seed(seed)

    if pruning_components == "all":
        pruning_components = "attn_heads+attn_layers+ffn_neurons+ffn_layers+hidden_state"
    pruning_components_list: list[str] = pruning_components.split("+")
    assert len(pruning_components), "Need to select at least one pruning method"

    do_prune_attention_heads = "attn_heads" in pruning_components_list
    do_prune_attention_heads_uniform = "attn_heads_uniform" in pruning_components_list
    do_prune_attention_layers = "attn_layers" in pruning_components_list
    do_prune_ffn_neurons = "ffn_neurons" in pruning_components_list
    do_prune_ffn_neurons_uniform = "ffn_neurons_uniform" in pruning_components_list
    do_prune_ffn_layers = "ffn_layers" in pruning_components_list
    do_prune_hidden_state = "hidden_state" in pruning_components_list

    # setup logging
    neptune_run = create_neptune_run(
        base_model=base_model,
        lib="our",
        pruning_ratio=pruning_ratio,
        pruning_components=pruning_components_list,
        calibration_dataset=pruning_dataset,
        calibration_batch_size=batch_size,
        calibration_num_samples=num_samples,
        calibration_how_to_collect=how_to_collect,
        calibration_how_to_average=how_to_average,
        calibration_how_to_overlap=how_to_overlap,
        save_model_as=save_model_as,
        extra_tags=["our"],
    )

    # Load the finetuned model and the corresponding tokenizer
    print(f"Loading model {base_model}...")
    config = AutoConfig.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, config=config)
    if IS_CUDA_AVAILABLE:
        model = model.to(device="cuda", non_blocking=True, dtype=torch.float16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    print(f"Original Model: {base_model} loaded")
    count_flops_macs_params(model, tokenizer, print_results=True)
    original_model_stats = measure_model_stats(model, print_results=False)

    # load dataset
    print(f"Loading dataset {pruning_dataset}...")
    calibration_dataset = get_tokenized_dataset(pruning_dataset, "train", tokenizer, num_samples, 128)
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    print(calibration_dataset)

    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"Calibration dataloader: {len(calibration_dataset)} samples")
    print("Dataset loaded")

    # print model with sample input
    print(model)

    print("-" * 80)

    # Collect components info
    if how_to_collect == "grads":
        components_info = collect_mask_gradients(model, calibration_dataloader)
    elif how_to_collect == "full_grads":
        components_info = collect_full_gradients(model, calibration_dataloader)
    elif how_to_collect == "activations":
        components_info = collect_activations(model, calibration_dataloader)
    elif how_to_collect == "weights":
        components_info = collect_weight_magnitudes(model)
    elif how_to_collect == "random":
        components_info = collect_random_numbers(model)
    else:
        assert False, f"Unknown how_to_collect: {how_to_collect}"

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
    relative_meta_importance = torch.nn.functional.softmax(
        components_importance.meta_importance.cpu().to(torch.float32), dim=0
    )

    # Log info and importance
    for name, value in components_info._asdict().items():
        neptune_run[f"pruning/{name}"].upload(File.as_pickle(tensor_to_list(value)))
    for name, value in components_importance._asdict().items():
        neptune_run[f"pruning/{name}"].upload(File.as_pickle(tensor_to_list(value)))

    # Print average importance
    print(f"Average importance for {how_to_collect}/{how_to_average}/{how_to_overlap}:")
    print_components_info_importance(components_importance)

    # Select pruning percent for each component type
    print(f"Select pruning percent for each component type, with {how_to_overlap} overlap")
    print(f"Base pruning ratio: {pruning_ratio}")
    if how_to_overlap == "fixed":
        # Prune same rate of specified components (e.g. 20% of attention heads, 20% of ffn neurons)
        attention_heads_pruning_ratio = (
            pruning_ratio if do_prune_attention_heads or do_prune_attention_heads_uniform else 0
        )
        attention_layers_pruning_ratio = pruning_ratio if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = pruning_ratio if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform else 0
        ffn_layers_pruning_ratio = pruning_ratio if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio if do_prune_hidden_state else 0
    elif how_to_overlap == "fixed_x2_x05":
        # Multiply attention heads deletion, decrease ffn neurons deletion
        attention_heads_pruning_ratio = (
            min(pruning_ratio * 2, 0.9) if do_prune_attention_heads or do_prune_attention_heads_uniform else 0
        )
        attention_layers_pruning_ratio = pruning_ratio if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = (
            min(pruning_ratio * 0.5, 0.9) if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform else 0
        )
        ffn_layers_pruning_ratio = pruning_ratio if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio if do_prune_hidden_state else 0
    elif how_to_overlap == "fixed_x05_x2":
        # Multiply ffn neurons deletion, decrease attention heads deletion
        attention_heads_pruning_ratio = (
            min(pruning_ratio * 0.5, 0.9) if do_prune_attention_heads or do_prune_attention_heads_uniform else 0
        )
        attention_layers_pruning_ratio = pruning_ratio if do_prune_attention_layers else 0
        ffn_neurons_pruning_ratio = (
            min(pruning_ratio * 2, 0.9) if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform else 0
        )
        ffn_layers_pruning_ratio = pruning_ratio if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = pruning_ratio if do_prune_hidden_state else 0
    elif how_to_overlap == "relative":
        # Use actual relative importance to calculate pruning ratio
        relative_importance = torch.tensor(
            [
                components_importance.attention_heads_importance.cpu().mean().item(),
                components_importance.attention_layers_importance.cpu().mean().item(),
                components_importance.ffn_neurons_importance.cpu().mean().item(),
                components_importance.ffn_layers_importance.cpu().mean().item(),
                components_importance.hidden_states_importance.cpu().mean().item(),
            ]
        )
        relative_pruning_coefficient = torch.softmax(-1 * relative_importance, dim=0) * relative_importance.shape[0]
        print("Relative importance", relative_importance)
        print("Relative pruning coefficient", relative_pruning_coefficient)
        attention_heads_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[0].item()
            if do_prune_attention_heads or do_prune_attention_heads_uniform
            else 0
        )
        attention_layers_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[1].item() if do_prune_attention_layers else 0
        )
        ffn_neurons_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[2].item()
            if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform
            else 0
        )
        ffn_layers_pruning_ratio = pruning_ratio * relative_pruning_coefficient[3].item() if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[4].item() if do_prune_hidden_state else 0
        )
    elif how_to_overlap == "relative_per_param":
        # Use actual relative importance to calculate pruning ratio
        # Use number of possible pruning parameters to calculate pruning ratio
        # adjust for number of params in the component
        _num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
        _single_head_size = config.hidden_size // config.num_attention_heads
        # k + v of the same size and q and output of the same size
        num_parameters_attention_head_group = (
            2 * _single_head_size * config.hidden_size + 2 * _single_head_size * _num_q_per_kv * config.hidden_size
        )
        num_parameters_attention_layer = num_parameters_attention_head_group * config.num_key_value_heads  # + bias
        # gate, input, output
        num_parameters_ffn_neuron = 3 * config.hidden_size
        num_parameters_ffn_layer = num_parameters_ffn_neuron * config.intermediate_size  # + bias
        # hidden state = emb + 2 norms, attention, ffn
        num_parameters_hidden_state = config.vocab_size + config.num_hidden_layers * (
            2
            + 2 * _single_head_size * config.num_key_value_heads
            + 2 * _single_head_size * config.num_attention_heads
            + 3 * config.intermediate_size
        )
        total_parameters = num_parameters_hidden_state * config.hidden_size + config.num_hidden_layers * (  # noqa: F841
            num_parameters_attention_layer + num_parameters_ffn_layer
        )

        relative_importance = torch.tensor(
            [
                components_importance.attention_heads_importance.cpu().mean().item()
                / num_parameters_attention_head_group,
                components_importance.attention_layers_importance.cpu().mean().item() / num_parameters_attention_layer,
                components_importance.ffn_neurons_importance.cpu().mean().item() / num_parameters_ffn_neuron,
                components_importance.ffn_layers_importance.cpu().mean().item() / num_parameters_ffn_layer,
                components_importance.hidden_states_importance.cpu().mean().item() / num_parameters_hidden_state,
            ]
        )
        relative_pruning_coefficient = torch.softmax(-1 * relative_importance, dim=0) * relative_importance.shape[0]
        print("Relative importance", relative_importance)
        print("Relative pruning coefficient", relative_pruning_coefficient)

        attention_heads_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[0].item()
            if do_prune_attention_heads or do_prune_attention_heads_uniform
            else 0
        )
        attention_layers_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[1].item() if do_prune_attention_layers else 0
        )
        ffn_neurons_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[2].item()
            if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform
            else 0
        )
        ffn_layers_pruning_ratio = pruning_ratio * relative_pruning_coefficient[3].item() if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[4].item() if do_prune_hidden_state else 0
        )
    elif how_to_overlap == "meta":
        # change the pruning ratio based on the relative importance of the components
        # e.g. given pruning ration of 20% and attention importance 2 times lower than ffn importance
        #   - prune 30% of attention and 10% of ffn
        relative_meta_pruning_coefficient = (
            torch.softmax(-1 * relative_meta_importance, dim=0) * relative_meta_importance.shape[0]
        )
        print("Relative meta importance", relative_meta_importance)
        print("Relative meta pruning coefficient", relative_meta_pruning_coefficient)
        attention_heads_pruning_ratio = (
            pruning_ratio * relative_meta_pruning_coefficient[0].item()
            if do_prune_attention_heads or do_prune_attention_heads_uniform
            else 0
        )
        attention_layers_pruning_ratio = (
            pruning_ratio * relative_meta_pruning_coefficient[1].item() if do_prune_attention_layers else 0
        )
        ffn_neurons_pruning_ratio = (
            pruning_ratio * relative_meta_pruning_coefficient[2].item()
            if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform
            else 0
        )
        ffn_layers_pruning_ratio = (
            pruning_ratio * relative_meta_pruning_coefficient[3].item() if do_prune_ffn_layers else 0
        )
        hidden_states_pruning_ratio = (
            pruning_ratio * relative_meta_pruning_coefficient[4].item() if do_prune_hidden_state else 0
        )
    else:
        assert False, f"Unknown how_to_overlap: {how_to_overlap}"
    print("  attention_heads_pruning_ratio", attention_heads_pruning_ratio)
    print("  attention_layers_pruning_ratio", attention_layers_pruning_ratio)
    print("  ffn_neurons_pruning_ratio", ffn_neurons_pruning_ratio)
    print("  ffn_layers_pruning_ratio", ffn_layers_pruning_ratio)
    print("  hidden_states_pruning_ratio", hidden_states_pruning_ratio)

    # Select components to prune
    attention_layers_to_prune = select_to_prune_attention_layers(
        components_importance.attention_layers_importance,
        attention_layers_pruning_ratio,
    )
    attention_heads_to_prune = select_to_prune_attention_heads(
        components_importance.attention_heads_importance,
        attention_heads_pruning_ratio,
        uniform_among_layers=do_prune_attention_heads_uniform,
        key_value_group_size=config.num_attention_heads // config.num_key_value_heads,
    )
    attention_heads_to_prune = {
        layer: heads for layer, heads in attention_heads_to_prune.items() if layer not in attention_layers_to_prune
    }

    ffn_layers_to_prune = select_to_prune_ffn_layers(
        components_importance.ffn_layers_importance,
        ffn_layers_pruning_ratio,
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
        components_importance.hidden_states_importance,
        hidden_states_pruning_ratio,
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
            print(
                f"> Layer {layer}: {num_heads_deleted} attention heads deleted ({num_grouped_heads_deleted} grouped heads deleted)"
            )
        total_percent_heads_deleted = total_num_heads_deleted / (config.num_attention_heads * config.num_hidden_layers)
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
            num_neurons_deleted = config.intermediate_size - model.model.layers[layer].mlp.up_proj.out_features
            total_num_neurons_deleted += num_neurons_deleted
            print(f"> Layer {layer}: {num_neurons_deleted} neurons deleted")
        total_percent_neurons_deleted = total_num_neurons_deleted / (
            config.intermediate_size * config.num_hidden_layers
        )
        print(
            f"Total: {total_percent_neurons_deleted * 100:.2f}% neurons deleted, "
            f"{(1 - total_percent_neurons_deleted) * 100:.2f}% remain"
        )

    print("-" * 80)
    pruned_model_stats = measure_model_stats(model, original_model_stats, print_results=True)
    neptune_run["pruning/stats"].upload(File.as_pickle(pruned_model_stats))
    neptune_run["pruning/original_stats"].upload(File.as_pickle(original_model_stats))

    if save_model_as:
        save_model_tokenizer(model, tokenizer, "results/" + save_model_as)

    # Log pruned model
    if evaluate_on:
        print("\n==================Evaluation after Pruning==================\n")
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            task_groups=evaluate_on,
            device="cuda" if IS_CUDA_AVAILABLE else "cpu",
        )
        neptune_run["evaluation"] = eval_results

    neptune_run.stop()


if __name__ == "__main__":
    typer.run(main)
