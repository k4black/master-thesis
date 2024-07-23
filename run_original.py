from __future__ import annotations

from typing import Optional

import torch
import typer

from adaptive_pruning.utils import measure_model_stats
from utils import (
    create_neptune_run,
    evaluate_model,
    load_llama_model,
    neptune_record_pruned_model,
    save_model_tokenizer,
    set_random_seed,
)


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(
    base_model: str = "huggyllama/llama-7b",
    attention_type: Optional[str] = "sdpa",
    pytorch_compile: bool = False,
    seed: int = 42,
    evaluate_on: Optional[str] = "perplexity+full+bias",
    save_model_as: Optional[str] = None,
    extra_tags: Optional[str] = None,
) -> None:
    set_random_seed(seed)

    # setup logging
    neptune_run = create_neptune_run(
        base_model=base_model,
        lib="original",
        pruning_ratio=0.0,
        pruning_components=[],
        num_iterations=0,
        calibration_dataset="",
        calibration_batch_size=0,
        calibration_num_samples=0,
        calibration_how_to_collect="",
        calibration_how_to_average="",
        calibration_how_to_overlap="",
        attention_type=attention_type,
        save_model_as=save_model_as,
        extra_tags=["original", "baseline", *extra_tags.split(",")] if extra_tags else ["original", "baseline"],
    )
    neptune_run["parameters/compile"] = pytorch_compile

    # Load the finetuned model and the corresponding tokenizer
    config, model, tokenizer = load_llama_model(
        base_model, attention_type=attention_type, device="cuda" if IS_CUDA_AVAILABLE else "cpu"
    )
    original_model_stats, original_model_size = measure_model_stats(model, tokenizer, print_results=False)

    # print model with sample input
    print(model)

    print("-" * 80)
    neptune_record_pruned_model(neptune_run, original_model_stats, original_model_size, None, None)

    if save_model_as:
        save_model_tokenizer(model, tokenizer, "results/" + save_model_as, neptune_run=neptune_run)

    if pytorch_compile:
        print("Compiling the model...")
        model = torch.compile(model)

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
