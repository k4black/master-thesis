import os
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import torch
import typer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from adaptive_pruning.utils import count_flops_macs_params, measure_model_stats
from utils import create_neptune_run, evaluate_model, neptune_record_pruned_model, save_model_tokenizer, set_random_seed


if typing.TYPE_CHECKING:
    from external.wanda.lib.prune import check_sparsity, prune_ablate, prune_magnitude, prune_sparsegpt, prune_wanda
else:
    # add external.llm_pruner to access LLMPruner
    os.sys.path.append((Path(__file__).parent / "external" / "wanda").as_posix())
    from lib.prune import check_sparsity, prune_ablate, prune_magnitude, prune_sparsegpt, prune_wanda


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_datasets_fix(*args: Any, **kwargs: Any) -> Any:
    """
    WANDA library loads C4 data legacy way causing error, this function fixes the issue.
    See https://github.com/huggingface/datasets/issues/6746

    We use unittest.mock.patch to replace the original function with this one.
    Patched function will ignore the "name" (second) argument.
    """
    if args[0] == "allenai/c4":
        return load_dataset(args[0], **kwargs)
    else:
        return load_dataset(*args, **kwargs)


@dataclass
class WandaLibArgs:
    """
    WANDA library requires Cli Args to be passed as input,
    to avoid modifying the library, we define dummy dataclass to pass the args in the library.
    Commented out args are not used in the library.
    """

    # model: int
    seed: int
    nsamples: int
    sparsity_ratio: float
    # sparsity_type: str
    prune_method: str
    # cache_dir: str
    use_variant: bool


def main(
    base_model: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    pruning_ratio: float = 0.5,
    num_samples: int = 128,
    sparsity_type: Optional[str] = "unstructured",  # ["unstructured", "4:8", "2:4"]
    prune_method: Optional[
        str
    ] = "wanda",  # ["magnitude", "wanda", "sparsegpt", "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"]
    seed: int = 42,
    evaluate_on: Optional[str] = "perplexity+full+bias",
    save_model_as: Optional[str] = None,
) -> None:
    set_random_seed(seed)
    custom_args = WandaLibArgs(
        seed=seed,
        nsamples=num_samples,
        sparsity_ratio=pruning_ratio,
        prune_method=prune_method,
        use_variant=False,
    )

    # setup logging
    neptune_run = create_neptune_run(
        base_model=base_model,
        lib="wanda",
        pruning_ratio=pruning_ratio,
        pruning_components=["weights"] if sparsity_type == "unstructured" else [f"weights-{sparsity_type}"],
        num_iterations=1,  # TBA
        calibration_dataset="c4",
        calibration_batch_size=1,
        calibration_num_samples=num_samples,
        calibration_how_to_collect=prune_method,
        calibration_how_to_average="none",
        calibration_how_to_overlap="",
        save_model_as=save_model_as,
        extra_tags=["wanda"],
    )

    device = torch.device("cuda:0") if IS_CUDA_AVAILABLE else torch.device("cpu")

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if sparsity_type != "unstructured":
        assert pruning_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, sparsity_type.split(":"))

    print(f"loading llm model {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=device
    )
    model.seqlen = model.config.max_position_embeddings
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    print(f"Original Model: {base_model} loaded")
    count_flops_macs_params(model, tokenizer, print_results=True)
    original_model_stats, original_model_size = measure_model_stats(model, tokenizer, print_results=False)

    if (
        "30b" in base_model or "65b" in base_model or "70b" in base_model
    ):  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if pruning_ratio != 0:
        # WARNING: Patch load_dataset to fix C4 loading issue, see load_datasets_fix docstring
        with patch("lib.data.load_dataset", new=load_datasets_fix):
            print("pruning starts")
            if prune_method == "wanda":
                prune_wanda(custom_args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif prune_method == "magnitude":
                prune_magnitude(custom_args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif prune_method == "sparsegpt":
                prune_sparsegpt(custom_args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif "ablate" in prune_method:
                prune_ablate(custom_args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################

    print("-" * 80)
    model.half()  # TODO: fix, next(model.parameters()).dtype float16, but error as full precision
    pruned_model_stats, pruned_model_size = measure_model_stats(
        model, tokenizer, original_model_stats, print_results=True
    )
    neptune_record_pruned_model(
        neptune_run, original_model_stats, original_model_size, pruned_model_stats, pruned_model_size
    )

    if save_model_as:
        save_model_tokenizer(model, tokenizer, "results/" + save_model_as, neptune_run=neptune_run)

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
