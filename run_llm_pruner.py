import copy
import os
import typing
from pathlib import Path
from typing import Optional

import torch
import typer
from transformers import LlamaTokenizer

from adaptive_pruning.utils import measure_original_model_stats, measure_pruned_model_stats


# from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention

if typing.TYPE_CHECKING:
    import external.llm_pruner.LLMPruner.torch_pruning as tp
    from external.llm_pruner.LLMPruner.datasets.example_samples import get_examples
    from external.llm_pruner.LLMPruner.models.hf_llama.modeling_llama import (
        LlamaAttention,
        LlamaForCausalLM,
        LlamaRMSNorm,
    )
    from external.llm_pruner.LLMPruner.pruner import hf_llama_pruner as llama_pruner
else:
    # add external.llm_pruner to access LLMPruner
    os.sys.path.append((Path(__file__).parent / "external" / "llm_pruner").as_posix())
    import LLMPruner.torch_pruning as tp
    from LLMPruner.pruner import hf_llama_pruner as llama_pruner
    from LLMPruner.datasets.example_samples import get_examples
    from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention

from utils import create_neptune_run, evaluate_model, save_model_tokenizer, set_random_seed


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(
    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    pruning_ratio: float = 0.5,
    pruner_type: str = "taylor",  # l1, l2, taylor
    taylor: str = "param_first",  # vectorize, param_second, param_first, param_mix
    max_seq_len: int = 128,
    channel_wise: bool = False,  # do channel wise or block wise pruning
    block_wise: bool = False,  # do block wise or layer wise pruning
    layer_wise: bool = False,  # do block wise or layer wise pruning
    keep_layers: int = 12,
    block_attention_layer_start: int = 4,
    block_attention_layer_end: int = 30,
    block_mlp_layer_start: int = 4,
    block_mlp_layer_end: int = 30,
    iterative_steps: int = 1,
    grouping_strategy: str = "sum",
    global_pruning: bool = False,
    num_examples: int = 10,  # number of examples to use for calibration
    seed: int = 42,
    evaluate_on: Optional[str] = "perplexity+full+toxicity",
    save_model_as: Optional[str] = None,
):
    set_random_seed(seed)

    if block_wise:
        pruning_components = ["attn-heads", "ffn-neurons"]
    elif channel_wise:
        pruning_components = ["hidden-state"]
    elif layer_wise:
        pruning_components = ["attn-layers", "ffn-layers"]
    else:
        raise ValueError("Please specify one of block_wise, channel_wise, layer_wise")

    # setup logging
    neptune_run = create_neptune_run(
        base_model=base_model,
        lib="llm-pruner",
        pruning_ratio=pruning_ratio,
        pruning_components=pruning_components,
        calibration_dataset="bookcorpus",
        calibration_batch_size=1,
        calibration_num_samples=num_examples,
        calibration_how_to_collect=pruner_type,
        calibration_how_to_average=taylor,
        calibration_how_to_overlap="none",
        save_model_as=save_model_as,
        extra_tags=["baseline"],
    )

    print(f"Loading model {base_model}...")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    if IS_CUDA_AVAILABLE:
        model.half()
        model.to("cuda")

    pruner_type = pruner_type.lower()
    assert pruner_type in ["random", "l2", "l1", "taylor"]

    for param in model.parameters():
        param.requires_grad_(True)
    original_model_stats = measure_original_model_stats(model, print_results=False)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Only for building the dependency graph.
    # Any input will be fine since the computation result are not taken into consideration.
    forward_prompts = torch.tensor(
        [
            [1, 306, 4658, 278, 6593, 310, 2834, 338],
            [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
        ]
    ).to(model.device)

    if pruner_type == "random":
        imp = tp.importance.RandomImportance()
    elif pruner_type == "l1":
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == "l2":
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == "taylor":
        imp = llama_pruner.TaylorImportance(group_reduction=grouping_strategy, taylor=taylor)
    else:
        raise NotImplementedError

    print(f"Use {pruner_type} pruner...")

    if block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": global_pruning,
            "iterative_steps": iterative_steps,
            "ch_sparsity": pruning_ratio,
            "ignored_layers": [],
            "channel_groups": {},
            "consecutive_groups": {layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers},
            "customized_pruners": {LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner},
            "root_module_types": None,
            "root_instances": [
                model.model.layers[i].self_attn.q_proj
                for i in range(block_attention_layer_start, block_attention_layer_end)
            ]
            + [model.model.layers[i].mlp.gate_proj for i in range(block_mlp_layer_start, block_mlp_layer_end)],
        }
        print(f"Pruning Attention Layer = {list(range(block_attention_layer_start, block_attention_layer_end))}")
        print(f"Pruning MLP Layer = {list(range(block_mlp_layer_start, block_mlp_layer_end))}")

        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()

        print("Start Pruning")
        for i in range(iterative_steps):
            print(f"Iter {i + 1}/{iterative_steps}")

            if pruner_type in ["taylor"]:
                example_prompts = get_examples("bookcorpus", tokenizer, num_examples, seq_len=64).to(model.device)
                print(f"Start Backwarding in iterative steps = {i}...")
                if taylor in ["param_mix", "param_second"]:
                    for j in range(num_examples):
                        print(f"Example {j + 1}/{num_examples}")
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        print(f"Loss = {loss}")
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / num_examples
                            if hasattr(module_param, "acc_grad"):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad

                loss = model(example_prompts, labels=example_prompts).loss
                print(f"Loss = {loss}")
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"After Iter {i + 1}/{iterative_steps}, #parameters: {after_pruning_parameters}")

            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if "weight" in name:
                module.grad = None

        del pruner

    elif channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": global_pruning,
            "iterative_steps": iterative_steps,
            "ch_sparsity": pruning_ratio,  # remove 50% channels
            "ignored_layers": [],
            # "round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                # layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                # LlamaAttention: llama_pruner.hf_attention_pruner,
            },
            "root_module_types": [LlamaRMSNorm, LlamaAttention],
        }

        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()

        print("Start Pruning")
        assert iterative_steps >= 1
        for i in range(iterative_steps):
            print(f"Iter {i + 1}/{iterative_steps}")

            if pruner_type in ["taylor"]:
                example_prompts = get_examples(
                    "bookcorpus",
                    tokenizer,
                    n_samples=num_examples,
                    seq_len=max_seq_len,
                ).to(model.device)
                print(f"Start Backwarding in iterative steps = {i}...")
                loss = model(example_prompts, labels=example_prompts).loss
                print(f"Loss = {loss}")
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(
                f"After Iter {i + 1}/{iterative_steps}, #parameters: {after_pruning_parameters}, "
                f"Ratio = {100.0 * after_pruning_parameters / before_pruning_parameters:.4f}%"
            )

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if "weight" in name:
                module.grad = None

        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()

        del pruner

    elif layer_wise:
        model.model.layers = model.model.layers[:keep_layers]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError

    print(
        f"#Param before: {before_pruning_parameters}, #Param after: {after_pruning_parameters}, "
        f"Ratio = {100.0 * after_pruning_parameters / before_pruning_parameters:.4f}%"
    )

    print("-" * 80)
    pruned_model_stats = measure_pruned_model_stats(model, original_model_stats, print_results=True)
    neptune_run["pruned_stats"] = pruned_model_stats

    if save_model_as:
        save_model_tokenizer(model, tokenizer, "results/" + save_model_as)

    print("-" * 64)
    print(model)

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if evaluate_on:
        print("\n==================Evaluation after Pruning==================\n")
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            task_groups=evaluate_on,
            device="cuda" if IS_CUDA_AVAILABLE else "cpu",
        )
        neptune_run["evaluation"] = eval_results


if __name__ == "__main__":
    typer.run(main)
