from __future__ import annotations

import logging
import random
from pathlib import Path

import lm_eval
import lm_eval.models.huggingface
import numpy as np
import torch
from datasets import load_dataset
from lm_eval.utils import make_table
from transformers import PreTrainedModel, PreTrainedTokenizer

from adaptive_pruning.utils import format_number, count_parameters, count_flops_macs_params


def set_random_seed(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'bookcorpus', split='train'
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def lm_eval_hf_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    name: str,
    tasks: list[str] = ["wikitext", "piqa", "boolq"],
    batch_size: int | str = "auto:4",
    device: str = "auto",
    dtype: str = "auto",
    logging_path: str | Path | None = None,
):
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

    # Measure flops
    flops, macs, params = count_flops_macs_params(model, tokenizer, print_results=False)

    # Print the results
    batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    print(f"{name=}, {batch_size=} ({batch_sizes}), {device=}, {dtype=}")
    # num_parameters = count_parameters(model)
    # print(f"Parameters: {format_number(num_parameters)} ({num_parameters})")
    print(f"FLOPs:  {format_number(flops)} ({flops})")
    print(f"MACs:   {format_number(macs)} ({macs})")
    print(f"Params: {format_number(params)} ({params})")
    print("Results:")
    print(make_table(results))
    # Save the results
    if logging_path:
        results_path = Path(logging_path) / f"{name}.json"
        results_path.write_text(results.json(indent=2))
