import time

import torch
from datasets import DatasetDict, load_dataset
from evaluate import load
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedModel, BatchEncoding


def load_glue_dataset(task_name: str, tokenizer: PreTrainedTokenizer) -> DatasetDict:
    text_1, text_2 = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }[task_name]

    # load dataset
    dataset = load_dataset("glue", task_name)

    # preprocess and tokenize
    def _tokenize(examples: dict) -> BatchEncoding:
        return tokenizer(
            examples[text_1],
            examples[text_2] if text_2 else None,
            padding=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

    tokenized_dataset = dataset.map(
        _tokenize,
        batched=True,
        load_from_cache_file=True,
        remove_columns=[col for col in dataset["train"].column_names if col not in ["label", "labels"]],
    )

    if task_name == "mnli":
        # rename splits for MNLI
        tokenized_dataset["validation"] = tokenized_dataset["validation_matched"]
        tokenized_dataset["test"] = tokenized_dataset["test_matched"]
        del tokenized_dataset["validation_matched"], tokenized_dataset["test_matched"]

    return tokenized_dataset


def measure_glue_metric(
    model: PreTrainedModel,
    dataloader: DataLoader,
    task_name: str,
    *,
    verbose: bool = False,
) -> tuple[float, float]:
    # load metric
    is_single_label = model.num_labels == 1
    metric = load("glue", task_name, trust_remote_code=True)
    # TODO: check if this is correct
    target_metric_name: str = {
        "cola": "matthews_correlation",
        "mnli": "accuracy",
        "mrpc": "accuracy",
        "qnli": "accuracy",
        "qqp": "accuracy",
        "rte": "accuracy",
        "sst2": "accuracy",
        "stsb": "pearson",
        "wnli": "accuracy",
    }[task_name]

    # evaluate
    model.eval()
    start_time = time.perf_counter()
    batch_iterator = tqdm(dataloader, total=len(dataloader), desc=f"Evaluating {task_name}", leave=False) if verbose else dataloader
    for batch in batch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        with torch.no_grad():
            outputs = model(**batch)

        if is_single_label:
            predictions = outputs.logits.squeeze()
        else:
            predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )
    elapsed_time = time.perf_counter() - start_time
    eval_results = metric.compute()
    return eval_results[target_metric_name], elapsed_time
