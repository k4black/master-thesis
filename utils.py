from __future__ import annotations

import gc
import logging
import random
import time
from pathlib import Path
from typing import Any, NamedTuple
from importlib.util import find_spec

import neptune
import lm_eval
import lm_eval.models.huggingface
import numpy as np
import torch
from datasets import load_dataset, Dataset
from lm_eval.utils import make_table
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, BatchEncoding, DataCollatorForLanguageModeling

from adaptive_pruning.utils import format_number, count_flops_macs_params


LM_EVAL_NAME_TO_TASKS = {
    "perplexity": ["wikitext"],
    "short": ["piqa", "boolq", "arc_easy"],
    "full": ["piqa", "boolq", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"],
    "gen": ["GSM8ka"],
    "toxicity": ["crows_pairs_english"],
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
    calibration_dataset: str,
    calibration_batch_size: int,
    calibration_num_samples: int,
    calibration_how_to_collect: str,  # gradients, activations, etc.
    calibration_how_to_average: str,  # mean, fisher_info, entropy, etc.
    calibration_how_to_overlap: str,  # fixed, relative, etc.
    save_model_as: str | None = None,
    finetuning: bool = False,
    finetuning_dataset: str | None = None,
    finetuning_batch_size: int | None = None,
    finetuning_num_samples: int | None = None,
    finetuning_learning_rate: float | None = None,
    finetuning_epochs: int | None = None,
    *,
    extra_tags: list[str] | None = None,
) -> neptune.Run:
    neptune_run = neptune.init_run(tags=[
        base_model,
        *(extra_tags or []),
    ])
    
    assert isinstance(pruning_components, list)

    neptune_run["parameters"] = {
        "base_model": base_model,
        "lib": "our",
        "pruning_ratio": pruning_ratio,
        "pruning_components": '+'.join(pruning_components),
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
) -> Dataset:
    if name == "bookcorpus":
        dataset_args = dict(path="bookcorpus")
        field = "text"
    elif name == "wikitext2":
        dataset_args = dict(path="Salesforce/wikitext", name="wikitext-2-v1")
        field = "text"
    else:
        raise NotImplementedError(f"Calibration dataset {name} is not supported.")
        
    dataset = load_dataset(**dataset_args, split=split, streaming=streaming and n_samples is not None, trust_remote_code=True)
    if n_samples:
        dataset = dataset.take(n_samples)
    if streaming:
        dataset = Dataset.from_list([{"text": ex[field]} for ex in dataset])
    if field != "text":
        dataset = dataset.rename_column(field, "text")
        
    def _tokenize(examples: dict) -> BatchEncoding:
        return tokenizer(
            examples['text'],
            padding=True,
            truncation=True,
            return_tensors='pt',
            # padding_side='left',
            max_length=seq_len or tokenizer.model_max_length,
        )

    tokenized_dataset = dataset.map(
        _tokenize,
        batched=True,
        load_from_cache_file=True,
        remove_columns=['text'],
    )

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
    dataset: str,
    split: str = 'test',
    n_samples: int | None = None,
    device: str = 'auto',
    repeat: int = 5,
) -> InferenceResult:
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Update cache
    model = model.to("cpu")
    model.eval()
    model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    
    # Move model to device
    model = model.to(device=device, dtype=torch.float32, memory_format=torch.contiguous_format)
    
    # Load dataset
    tokenized_dataset = get_tokenized_dataset(
        name=dataset,
        split=split,
        tokenizer=tokenizer,
        n_samples=n_samples,
        seq_len=None,
    )
    
    # Init cuda events loggers
    start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # For each repetition 
    timings_average, timings_std = [], []
    is_cuda_available = torch.cuda.is_available()
    model.eval()
    with torch.no_grad():
        for _ in range(repeat):
            repeat_timings = []
            # Warm-up GPU
            for i in range(32):
                inputs = {k: v.to(device) for k, v in tokenized_dataset[i].items()}
                _ = model(**inputs)
                if is_cuda_available:
                    torch.cuda.synchronize()
                
            # Measure inference time
            for i in range(len(tokenized_dataset)):
                inputs = {k: v.to(device) for k, v in tokenized_dataset[i].items()}
                start_time.record()
                _ = model(**inputs)
                end_time.record()
                if is_cuda_available:
                    torch.cuda.synchronize()
                operation_time_s = start_time.elapsed_time(end_time) / 1000  # to seconds
                repeat_timings.append(operation_time_s)
            
            timings_average.append(np.mean(repeat_timings))
            timings_std.append(np.std(repeat_timings))
    
    return InferenceResult(
        time_average=np.mean(timings_average),
        time_std=np.mean(timings_std),
        n_samples=len(tokenized_dataset),
        time_per_sample_average=np.mean(timings_average) / len(tokenized_dataset),
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
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
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
        print_results=False,
    )
    batch_sizes = ",".join(map(str, lm_eval_results["config"]["batch_sizes"]))
    # print('lm_eval_results["results"]', lm_eval_results["results"])
    short_lm_eval_results = {
        task: [v for k, v in task_results.items() if 'stderr' not in k and 'alias' not in k and ('acc' in k or 'f1' in k or 'word_perplexity' in k)][0]
        for task, task_results in lm_eval_results["results"].items()
    }
    # print('short_lm_eval_results', short_lm_eval_results)

    # Measure inference time
    inference_result: InferenceResult = measure_inference_time(
        model=model, 
        tokenizer=tokenizer,
        dataset=inference_dataset, 
        split=inference_dataset_split,
        device=device, 
    )

    # Print the results
    print(f">> {batch_size=} ({batch_sizes}), {device=}, {dtype=}")
    flops, macs, params = count_flops_macs_params(model, tokenizer, print_results=True)
    print(make_table(lm_eval_results))
    for task, result in short_lm_eval_results.items():
        print(f"{task:>10}: {result:.4f}")
    print(f"Inference time: {inference_result.time_average:.2f}s ±{inference_result.time_std:.2f} for {inference_result.n_samples} samples")

    return {
        # TODO: fix int32 overflow in neptune
        "flops": float(flops),
        "macs": float(macs),
        "params": float(params),
        "inference_dataset": inference_dataset + "/" + inference_dataset_split,
        "inference_time_average": inference_result.time_average,
        "inference_time_std": inference_result.time_std,
        "inference_num_samples": inference_result.n_samples,
        "inference_time_per_sample_average": inference_result.time_per_sample_average,
        **short_lm_eval_results,
    }


def save_model_tokenizer(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, path: str | Path) -> None:
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving model to {model_path}...")
    
    model.half()
    torch.save(
        {
            'model': model,
            'tokenizer': tokenizer,
        }, 
        model_path,
    )
    

def load_model_tokenizer(path: str | Path) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_path = Path(path)
    
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    
    model.eval()
    model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    
    return model
