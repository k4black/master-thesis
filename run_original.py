from __future__ import annotations

from typing import Optional

import torch
import typer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from adaptive_pruning.utils import count_flops_macs_params, measure_original_model_stats, measure_pruned_model_stats
from utils import create_neptune_run, evaluate_model, set_random_seed


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(
    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    seed: int = 42,
    evaluate_on: Optional[str] = "perplexity+full+toxicity",
) -> None:
    set_random_seed(seed)

    # setup logging
    neptune_run = create_neptune_run(
        base_model=base_model,
        lib="original",
        pruning_ratio=0,
        pruning_components=[],
        calibration_dataset="",
        calibration_batch_size=0,
        calibration_num_samples=0,
        calibration_how_to_collect="",
        calibration_how_to_average="",
        calibration_how_to_overlap="",
        save_model_as=None,
        extra_tags=[],
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
    original_model_stats = measure_original_model_stats(model, print_results=False)

    # print model with sample input
    print(model)

    print("-" * 80)
    pruned_model_stats = measure_pruned_model_stats(model, original_model_stats, print_results=True)
    neptune_run["pruned_stats"] = pruned_model_stats

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
