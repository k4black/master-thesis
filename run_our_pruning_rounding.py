from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import typer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from adaptive_pruning.pruning import prune_ffn_neurons
from adaptive_pruning.utils import count_flops_macs_params, measure_model_stats
from utils import create_neptune_run, measure_inference_time, neptune_record_pruned_model, set_random_seed


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(
    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    is_uniform: bool = False,
    pruning_ratio: float = 0.2,
    round_to: Optional[int] = None,
    seed: int = 42,
) -> None:
    set_random_seed(seed)

    # setup logging
    neptune_run = create_neptune_run(
        base_model=base_model,
        lib="-",
        pruning_ratio=pruning_ratio,
        pruning_components=["ffn-neurons"],
        num_iterations=0,
        calibration_dataset="",
        calibration_batch_size=0,
        calibration_num_samples=0,
        calibration_how_to_collect="random",
        calibration_how_to_average=f"round_to={round_to}",
        calibration_how_to_overlap=f"is_uniform={is_uniform}",
        save_model_as=None,
        extra_tags=["compare_pruning_rounding"],
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
    original_model_stats, original_model_size = measure_model_stats(model, tokenizer, print_results=False)

    # print model with sample input
    print(model)

    if pruning_ratio > 0:
        # Caclulate the number of neurons to prune
        if is_uniform:
            ratios_to_prune = {i: pruning_ratio for i in range(model.config.num_hidden_layers)}
        else:
            random_pruning_rates = np.random.normal(pruning_ratio, 0.05, model.config.num_hidden_layers)
            random_pruning_rates = random_pruning_rates + (np.mean(random_pruning_rates) - pruning_ratio)
            random_pruning_rates = np.clip(random_pruning_rates, 0.0, 0.9)

            ratios_to_prune = {i: random_pruning_rates[i] for i in range(model.config.num_hidden_layers)}
        print(f"Pruning ratios: {ratios_to_prune}")

        num_of_neurons_to_prune = {
            i: int(ratios_to_prune[i] * model.config.intermediate_size) for i in range(model.config.num_hidden_layers)
        }
        if round_to is not None:
            num_of_neurons_to_prune = {
                i: round(num_of_neurons_to_prune[i] / round_to) * round_to
                for i in range(model.config.num_hidden_layers)
            }
        print(f"Number of neurons to prune (with {round_to=}): {num_of_neurons_to_prune}")

        # randomly select neurons to prune as list with values from 0 to num_ffn_neurons
        neurons_to_prune = {
            layer: list(np.random.choice(model.config.intermediate_size, num_neurons_to_prune, replace=False))
            for layer, num_neurons_to_prune in num_of_neurons_to_prune.items()
        }
        # print(f"Neurons to prune (layer to len): {neurons_to_prune.items(), lambda x: (x[0], len(x[1])))}")

        # Prune the model
        prune_ffn_neurons(model, neurons_to_prune)

    print("-" * 80)
    model.half()  # TODO: fix this in pruning code, keep same dtype as before
    pruned_model_stats, pruned_model_size = measure_model_stats(
        model, tokenizer, original_model_stats, print_results=True
    )
    neptune_record_pruned_model(
        neptune_run, original_model_stats, original_model_size, pruned_model_stats, pruned_model_size
    )

    # Measure inference time
    inference_result = measure_inference_time(
        model=model,
        tokenizer=tokenizer,
        dataset="wikitext2",
        split="test",
        device="cuda" if IS_CUDA_AVAILABLE else "cpu",
        repeat=3,
    )

    print(
        f"Inference time: {inference_result.time_average:.2f}s ±{inference_result.time_std:.2f} "
        f"on {inference_result.n_samples} samples ({3} repetitions)"
    )

    neptune_run["evaluation"] = {
        "inference_dataset": "wikitext2/test",
        "inference_time_average": inference_result.time_average,
        "inference_time_std": inference_result.time_std,
        "inference_num_samples": inference_result.n_samples,
        "inference_time_per_sample_average": inference_result.time_per_sample_average,
    }

    neptune_run.stop()


if __name__ == "__main__":
    typer.run(main)
