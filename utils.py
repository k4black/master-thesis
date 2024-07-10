from __future__ import annotations

import gc
import logging
import random
from pathlib import Path
from typing import Any, NamedTuple

import lm_eval
import lm_eval.models.huggingface
import neptune
import numpy as np
import torch
from datasets import Dataset, load_dataset
from lm_eval.utils import make_table
from neptune.types import File
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedModel, PreTrainedTokenizer


LM_EVAL_NAME_TO_TASKS = {
    "perplexity": ["wikitext"],
    "short": ["piqa", "boolq", "arc_easy"],
    "full": ["piqa", "boolq", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"],
    "extra": ["pile_10k", "gsm8k", "gsm8k_cot", "toxigen"],  # "gsm8k_cot" and "toxigen" to long?
    "bias": [
        "truthfulqa_mc1",
        "crows_pairs_english",
        "crows_pairs_english_age",
        "crows_pairs_english_autre",
        "crows_pairs_english_disability",
        "crows_pairs_english_gender",
        "crows_pairs_english_nationality",
        "crows_pairs_english_physical_appearance",
        "crows_pairs_english_race_color",
        "crows_pairs_english_religion",
        "crows_pairs_english_sexual_orientation",
        "crows_pairs_english_socioeconomic",
    ],
}
# TODO: add crows_pairs (crows_pairs_english),realtoxicityprompts,toxigen,truthfulqa,model_written_evals


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_neptune_run(
    base_model: str,
    lib: str,
    pruning_ratio: float,
    pruning_components: list[str],
    num_iterations: int,
    calibration_dataset: str,
    calibration_batch_size: int,
    calibration_num_samples: int,
    calibration_how_to_collect: str,  # gradients, activations, etc.
    calibration_how_to_average: str,  # mean, fisher_info, entropy, etc.
    calibration_how_to_overlap: str,  # fixed, relative, etc.
    save_model_as: str | None = None,
    pruning_round_to: int | None = None,
    finetuning: bool = False,
    finetuning_dataset: str | None = None,
    finetuning_batch_size: int | None = None,
    finetuning_num_samples: int | None = None,
    finetuning_learning_rate: float | None = None,
    finetuning_epochs: int | None = None,
    *,
    extra_tags: list[str] | None = None,
) -> neptune.Run:
    neptune_run = neptune.init_run(
        tags=[
            base_model,
            *(extra_tags or []),
        ]
    )

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    device_gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    device_gpu_memory_gb = device_gpu_memory / 1024**3

    assert isinstance(pruning_components, list)

    neptune_run["monitoring/device"] = device_name
    neptune_run["monitoring/device_memory"] = fix_neptune_overflow_recursively(device_gpu_memory)
    neptune_run["monitoring/device_memory_gb"] = device_gpu_memory_gb

    neptune_run["parameters"] = {
        "base_model": base_model,
        "lib": lib,
        "pruning_ratio": pruning_ratio,
        "pruning_components": "+".join(pruning_components) if pruning_components else "",
        "pruning_round_to": pruning_round_to or 1,
        "num_iterations": num_iterations,
        "calibration_dataset": calibration_dataset,
        "calibration_batch_size": calibration_batch_size,
        "calibration_num_samples": calibration_num_samples,
        "calibration_how_to_collect": calibration_how_to_collect,
        "calibration_how_to_average": calibration_how_to_average,
        "calibration_how_to_overlap": calibration_how_to_overlap,
        "save_model_as": save_model_as,
        # "finetuning": finetuning,
        # "finetuning_dataset": finetuning_dataset,
        # "finetuning_batch_size": finetuning_batch_size,
        # "finetuning_num_samples": finetuning_num_samples,
        # "finetuning_learning_rate": finetuning_learning_rate,
        # "finetuning_epochs": finetuning_epochs,
    }

    return neptune_run


def get_tokenized_dataset(
    name: str,
    split: str,
    tokenizer: PreTrainedTokenizer,
    n_samples: int | None = None,
    seq_len: int | None = None,
    streaming: bool = True,
    drop_empty_strings: bool = True,
    padding: str | bool = "longest",
) -> Dataset:
    if name == "bookcorpus":
        dataset_args = dict(path="bookcorpus")
        field = "text"
    elif name == "wikitext2":
        dataset_args = dict(path="Salesforce/wikitext", name="wikitext-2-v1")
        field = "text"
    elif name == "c4":
        dataset_args = dict(path="allenai/c4", name="en")
        field = "text"
    else:
        raise NotImplementedError(f"Calibration dataset {name} is not supported.")

    dataset = load_dataset(
        **dataset_args,
        split=split,
        streaming=streaming and n_samples is not None,
        trust_remote_code=True,
    )
    if drop_empty_strings:
        dataset = dataset.filter(lambda x: x[field] != "")
    if n_samples:
        dataset = dataset.take(n_samples)
    if streaming:
        dataset = Dataset.from_list([{"text": ex[field]} for ex in dataset])
    if field != "text":
        dataset = dataset.rename_column(field, "text")

    MAX_INT_32 = 2147483647
    tokenizer.model_max_length = min(MAX_INT_32, tokenizer.model_max_length)

    def _tokenize(examples: dict) -> BatchEncoding:
        return tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            # return_tensors="pt",
            # padding_side='left',
            max_length=seq_len or tokenizer.model_max_length,
        )

    tokenized_dataset = dataset.map(
        _tokenize,
        batched=True,
        load_from_cache_file=True,
        remove_columns=["text"],
    )
    # pt_columns = ["input_ids", "token_type_ids", "attention_mask", "label"]
    # tokenized_dataset.set_format(type="torch", columns=[i for i in pt_columns if i in tokenized_dataset.column_names])

    return tokenized_dataset


def lm_eval_hf_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task_groups: str,  # + split, see LM_EVAL_NAME_TO_TASKS
    batch_size: int | str = "auto:4",
    device: str = "auto",
    dtype: str = "auto",
    print_results: bool = True,
) -> dict[str, Any]:
    try:
        batch_size = int(batch_size)
    except ValueError:
        pass
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto" and device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32
    model = model.to(device, memory_format=torch.contiguous_format, dtype=dtype)

    # get tasks list from name
    tasks = []
    for task_group_name in task_groups.split("+"):
        if task_group_name not in LM_EVAL_NAME_TO_TASKS:
            raise ValueError(f"Task {task_group_name} is not supported.")
        tasks.extend(LM_EVAL_NAME_TO_TASKS[task_group_name])
    print(f"Running groups: {task_groups.split('+')}")
    print(f"-> Tasks: {tasks}")

    # setup lm_eval logging level to ERROR
    logger = logging.getLogger("lm-eval")
    logger.setLevel(logging.ERROR)

    # check if vllm is installed
    # if find_spec("vllm") is None:

    lm_obj = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        dtype=dtype,
        parallelize=False,
        device=device,
        batch_size=batch_size,
    )

    # trust remote code for HF datasets, new version requires this to work
    # https://github.com/EleutherAI/lm-evaluation-harness/blob/d855d0baf8576296e790d0c9477b40a710d28e67/lm_eval/__main__.py#L358
    import datasets

    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        log_samples=False,
        verbosity="ERROR",
        cache_requests=True,
        batch_size=batch_size,
        device="auto",
    )

    if print_results:
        print("Results:")
        print(make_table(results))

    return results


class InferenceResult(NamedTuple):
    time_average: float
    time_std: float
    n_samples: int
    time_per_sample_average: float


def measure_inference_time(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: str = "wikitext2",
    split: str = "test",
    n_samples: int | None = None,
    device: str = "auto",
    repeat: int = 3,
) -> InferenceResult:
    print(f"Measure inference time on {dataset}/{split} select {n_samples}")
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Update cache
    model = model.to("cpu")
    model.eval()
    model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # Move model to device
    model = model.to(device=device, dtype=torch.float16, memory_format=torch.contiguous_format)

    # Load dataset
    tokenized_dataset = get_tokenized_dataset(
        name=dataset,
        split=split,
        tokenizer=tokenizer,
        n_samples=n_samples,
        seq_len=None,
        padding=False,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest", return_tensors="pt")
    tokenized_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
        collate_fn=data_collator,
    )
    # print('sample dataset', tokenized_dataset[0])
    # for i in tokenized_dataloader:
    #     print('sample dataloader', i)
    #     break

    # Init cuda events loggers
    start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # For each repetition
    timings_history = []
    is_cuda_available = torch.cuda.is_available()
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(repeat), desc="repetitions"):
            repeat_timings = []
            # Warm-up GPU
            for i, batch in enumerate(tokenized_dataloader):
                if i > 32:
                    break
                batch = {k: v.to(model.device) for k, v in batch.items()}

                _ = model(**batch)
                if is_cuda_available:
                    torch.cuda.synchronize()

            # Measure inference time
            for i, batch in enumerate(tokenized_dataloader):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                start_time.record()
                _ = model(**batch)
                end_time.record()
                if is_cuda_available:
                    torch.cuda.synchronize()
                operation_time_s = start_time.elapsed_time(end_time) / 1000  # to seconds
                repeat_timings.append(operation_time_s)

            timings_history.append(np.sum(repeat_timings))  # sum for all dataset

    return InferenceResult(
        time_average=np.mean(timings_history),  # mean across repeats
        time_std=np.std(timings_history),  # std across repeats
        n_samples=len(tokenized_dataset),
        time_per_sample_average=np.mean(timings_history) / len(tokenized_dataset),
    )


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task_groups: str,  # + split, see LM_EVAL_NAME_TO_TASKS
    inference_dataset: str = "wikitext2",
    inference_dataset_split: str = "test",
    batch_size: int | str = "auto:4",
    device: str = "auto",
    dtype: str = "auto",
) -> dict[str, Any]:
    # Reload model to gpu, update gpu cache
    model = model.to("cpu")
    model = model.to(torch.float32)
    model.eval()
    model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    # TODO: ? save model to tmp file, then reload it
    # tmp_model_path = Path(f"results/{name}.pt")
    # tmp_model_path.parent.mkdir(parents=True, exist_ok=True)
    # torch.save({
    #     'model': model,
    # }, tmp_model_path)
    # model = torch.load(tmp_model_path, map_location='cpu')['model']

    # Evaluate the model with lm_eval
    lm_eval_results = lm_eval_hf_model(
        model,
        tokenizer,
        task_groups=task_groups,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        print_results=True,
    )
    batch_sizes = ",".join(map(str, lm_eval_results["config"]["batch_sizes"]))
    for task, task_results in lm_eval_results["results"].items():
        print(f"{task}: {task_results}")
    short_lm_eval_results = {
        task: [
            v
            for k, v in task_results.items()
            if "stderr" not in k
            and "alias" not in k
            and ("acc," in k or "f1," in k or "word_perplexity," in k or "pct_stereotype," in k or "exact_match," in k)
        ][0]
        for task, task_results in lm_eval_results["results"].items()
    }
    print("short_lm_eval_results", short_lm_eval_results)
    # add average across all present tasks in short and full LM_EVAL_NAME_TO_TASKS (None if some task is missing)
    short_lm_eval_results["short_average"] = np.mean(
        [short_lm_eval_results.get(task, np.nan) for task in LM_EVAL_NAME_TO_TASKS["short"]]
    )
    short_lm_eval_results["full_average"] = np.mean(
        [short_lm_eval_results.get(task, np.nan) for task in LM_EVAL_NAME_TO_TASKS["full"]]
    )

    # Measure inference time
    inference_result: InferenceResult = measure_inference_time(
        model=model,
        tokenizer=tokenizer,
        dataset=inference_dataset,
        split=inference_dataset_split,
        device=device,
        repeat=3,
    )

    # Print the results
    print(f">> {batch_size=} ({batch_sizes}), {device=}, {dtype=}")
    print(make_table(lm_eval_results))
    for task, result in short_lm_eval_results.items():
        print(f"{task:>10}: {result:.4f}")
    print(
        f"Inference time: {inference_result.time_average:.2f}s ±{inference_result.time_std:.2f} "
        f"on {inference_result.n_samples} samples ({3} repetitions)"
    )

    return {
        "inference_dataset": inference_dataset + "/" + inference_dataset_split,
        "inference_time_average": inference_result.time_average,
        "inference_time_std": inference_result.time_std,
        "inference_num_samples": inference_result.n_samples,
        "inference_time_per_sample_average": inference_result.time_per_sample_average,
        **short_lm_eval_results,
    }


def save_model_tokenizer(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, path: str | Path, neptune_run: neptune.Run | None = None
) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {model_path}...")

    model.half()
    torch.save(
        {
            "model": model,
            "tokenizer": tokenizer,
            "neptune_run_id": str(neptune_run._sys_id) if neptune_run else None,
        },
        model_path,
    )


def load_model_tokenizer(path: str | Path) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_path = Path(path)

    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path)
    model = checkpoint["model"]
    tokenizer = checkpoint["tokenizer"]

    model.eval()
    model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    return model, tokenizer


def check_and_convert_neptune_overflow(value: Any) -> Any:
    """
    Check if the value is within the 32-bit signed integer range.
    If not, convert it to a float.
    """
    INT32_MIN = -(2**31)
    INT32_MAX = 2**31 - 1

    if isinstance(value, int):
        if value < INT32_MIN or value > INT32_MAX:
            return float(value)
    return value


def fix_neptune_overflow_recursively(data: dict[str, Any] | list[Any] | Any) -> Any:
    if isinstance(data, dict):
        return {k: fix_neptune_overflow_recursively(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [fix_neptune_overflow_recursively(item) for item in data]
    else:
        return check_and_convert_neptune_overflow(data)


def neptune_record_pruned_model(
    neptune_run: neptune.Run,
    original_model_stats: dict,
    original_model_size: dict,
    pruned_model_stats: dict | None,
    pruned_model_size: dict | None,
) -> None:
    if pruned_model_stats:
        neptune_run["pruning/pruned_stats"].upload(File.as_pickle(pruned_model_stats))
    neptune_run["pruning/original_stats"].upload(File.as_pickle(original_model_stats))

    if pruned_model_size:
        neptune_run["pruning/pruned_size"] = fix_neptune_overflow_recursively(pruned_model_size)
    neptune_run["pruning/original_size"] = fix_neptune_overflow_recursively(original_model_size)

    if pruned_model_size:
        # percent of original params
        neptune_run["pruning/percent_left"] = pruned_model_size["params"] / original_model_size["params"] * 100
        # non zero percent of original params
        neptune_run["pruning/percent_nonzero_left"] = (
            (pruned_model_size["params"] - pruned_model_size["zero_params"]) / original_model_size["params"] * 100
        )
    else:
        neptune_run["pruning/percent_left"] = 100.0
        neptune_run["pruning/percent_nonzero_left"] = 100.0
