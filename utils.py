from __future__ import annotations

import gc
import logging
import random
import time
from pathlib import Path
from typing import Any

import lm_eval
import lm_eval.models.huggingface
import numpy as np
import torch
from datasets import load_dataset, Dataset
from lm_eval.utils import make_table
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, BatchEncoding, DataCollatorForLanguageModeling

from adaptive_pruning.utils import format_number, count_parameters, count_flops_macs_params


def set_random_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_bookcorpus(
        tokenizer:
        PreTrainedTokenizer,
        n_samples: int,
        seq_len: int | None = None,
        streaming: bool = True,
) -> Dataset:
    if streaming:
        dataset = load_dataset(
            'bookcorpus', split='train', streaming=True,
        )
        dataset = dataset.take(n_samples)
        dataset = Dataset.from_list([{'text': ex['text']} for ex in dataset])
    else:
        dataset = load_dataset(
            'bookcorpus', split='train'
        )
        dataset = dataset.select(range(n_samples))

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
    name: str,
    tasks: list[str] = ["wikitext", "piqa", "boolq", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"],
    batch_size: int | str = "auto:4",
    device: str = "auto",
    dtype: str = "auto",
    logging_path: str | Path | None = None,
    print_results: bool = True,
) -> dict[str, Any]:
    try:
        batch_size = int(batch_size)
    except ValueError:
        pass
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        model = model.to(torch.float32)
    model = model.to(device)

    # setup lm_eval logging level to ERROR
    logger = logging.getLogger("lm-eval")
    logger.setLevel(logging.ERROR)

    lm_obj = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        dtype=dtype,
        parallelize=False,
        device=device,
        batch_size=batch_size,
    )

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        log_samples=False,
        verbosity="ERROR",
        cache_requests=True,
        batch_size=batch_size,
        device=device,
    )

    if print_results:
        print("Results:")
        print(make_table(results))

    # Save the results
    if logging_path:
        results_path = Path(logging_path) / f"{name}.json"
        results_path.write_text(results.json(indent=2))

    return results


def measure_inference_time(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inference_dataset: Dataset | str,
    batch_size: int | str = 8,
    device: str = 'auto',
    dataset_size: int = 10000,
    repeat: int = 3,
) -> tuple[float, int, float]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        model = model.to(torch.float32)
    model = model.to(device)

    if isinstance(inference_dataset, str):
        if inference_dataset == "bookcorpus":
            inference_dataset = get_bookcorpus(tokenizer, dataset_size, 128)
        else:
            inference_dataset = load_dataset(inference_dataset, split='train')
            inference_dataset = inference_dataset.select(range(dataset_size))
            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=True, padding=False)
            inference_dataset = inference_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    elif isinstance(inference_dataset, Dataset):
        inference_dataset = inference_dataset.select(range(dataset_size))
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=False)
        inference_dataset = inference_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    else:
        raise ValueError(f"Unsupported type for inference_dataset: {type(inference_dataset)}")

    try:
        batch_size = int(batch_size)
    except ValueError:
        pass

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(inference_dataset, batch_size=batch_size, collate_fn=data_collator, pin_memory=True)

    # Measure inference time
    model.eval()
    total_inference_time_s = 0
    with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=torch.float16 if device == "cuda" else torch.float32):
        with torch.no_grad():
            for _ in range(repeat):
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                for batch in tqdm(dataloader, total=len(dataloader), desc=f"Inference", unit="batch", leave=True):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    model(**inputs)
                end_time.record()
                torch.cuda.synchronize()
                total_inference_time_s += start_time.elapsed_time(end_time) / 1000  # in seconds

    inference_time = total_inference_time_s / repeat
    samples_per_sec = len(inference_dataset) / inference_time

    return inference_time, len(inference_dataset), samples_per_sec


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inference_dataset: Dataset | str,
    name: str,
    tasks: list[str] = ["wikitext", "piqa", "boolq", "arc_easy"],
    batch_size: int | str = "auto:4",
    device: str = "auto",
    dtype: str = "auto",
    logging_path: str | Path | None = None,
) -> dict[str, Any]:
    # Reload model to gpu, update gpu cache
    model = model.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    for param in model.parameters():
        param.requires_grad_(False)

    # save model to tmp file, then reload it
    tmp_model_path = Path(f"results/{name}.pt")
    tmp_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': model,
    }, tmp_model_path)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(10)
    model = torch.load(tmp_model_path, map_location='cpu')['model']
    for param in model.parameters():
        param.requires_grad_(False)

    # Reload model to gpu, update gpu cache
    gc.collect()
    torch.cuda.empty_cache()

    if torch.cuda.is_available() and (device == "cuda" or device == "auto"):
        model = model.to(device="cuda", dtype=torch.float16)
    else:
        model = model.to(device="cpu", dtype=torch.float32)
    model.eval()

    # Measure flops
    flops, macs, params = count_flops_macs_params(model, tokenizer, print_results=False)

    # Evaluate the model with lm_eval
    lm_eval_results = lm_eval_hf_model(
        model, tokenizer, name, tasks=tasks, batch_size=batch_size, device=device, dtype=dtype, logging_path=logging_path, print_results=False,
    )
    batch_sizes = ",".join(map(str, lm_eval_results["config"]["batch_sizes"]))
    print('lm_eval_results["results"]', lm_eval_results["results"])
    short_lm_eval_results = {
        task: [v for k, v in task_results.items() if 'stderr' not in k and 'alias' not in k and ('acc' in k or 'f1' in k or 'word_perplexity' in k)][0]
        for task, task_results in lm_eval_results["results"].items()
    }
    print('short_lm_eval_results', short_lm_eval_results)

    # Measure inference time
    inference_batch_size = 8 if isinstance(batch_size, str) else batch_size
    inference_time, inference_num_samples, inference_samples_per_sec = measure_inference_time(model, tokenizer, inference_dataset, inference_batch_size, device)

    # Print the results
    print(f">> {name=}\n  {batch_size=} ({batch_sizes}), {device=}, {dtype=}")
    print(f"FLOPs:  {format_number(flops)} ({flops})")
    print(f"MACs:   {format_number(macs)} ({macs})")
    print(f"Params: {format_number(params)} ({params})")
    print(make_table(lm_eval_results))
    for task, result in short_lm_eval_results.items():
        print(f"{task:>10}: {result:.4f}")
    print(f"Inference time: {inference_time:.2f}s for {inference_num_samples} samples ({inference_samples_per_sec:.2f} samples/s)")

    return {
        # TODO: fix int32 overflow in neptune
        "flops": float(flops),
        "macs": float(macs),
        "params": float(params),
        "inference_time": inference_time,
        "inference_num_samples": inference_num_samples,
        "inference_samples_per_sec": inference_samples_per_sec,
        **short_lm_eval_results,
    }
