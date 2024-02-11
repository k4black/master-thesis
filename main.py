from __future__ import annotations

import time

import click
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
)
from tokenizers import Tokenizer
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from evaluate import load
from tqdm import tqdm


IS_TPU_AVAILABLE = torch.cuda.is_available() and "TPU" in torch.cuda.get_device_name(0)
IS_CUDA_AVAILABLE = torch.cuda.is_available() and not IS_TPU_AVAILABLE
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE
# IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.backends.cuda.is_bf16_supported
print(f"device: {torch.cuda.get_device_name(0)}")
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")
print(f"FP16_AVAILABLE: {IS_FP16_AVAILABLE}")
# print(f"BF16_AVAILABLE: {IS_BF16_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_glue_dataset(task_name: str, tokenizer: Tokenizer) -> DatasetDict:
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
    def _tokenize(examples: dict) -> dict:
        return tokenizer(
            examples[text_1],
            examples[text_2] if text_2 else None,
            padding=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

    dataset = dataset.map(
        _tokenize,
        batched=True,
        load_from_cache_file=True,
        remove_columns=[
            col
            for col in dataset["train"].column_names
            if col not in ["label", "labels"]
        ],
    )

    if task_name == "mnli":
        # rename splits for MNLI
        dataset["validation"] = dataset["validation_matched"]
        dataset["test"] = dataset["test_matched"]
        del dataset["validation_matched"], dataset["test_matched"]

    return dataset


def measure_metric(
    model: PreTrainedModel,
    head_mask: torch.Tensor | None,
    dataloader: DataLoader,
    task_name: str,
) -> tuple[float, float]:
    # load metric
    is_single_label = model.num_labels == 1
    metric = load("glue", task_name, trust_remote_code=True)
    target_metric_name = {
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
    for batch in tqdm(
        dataloader, total=len(dataloader), desc=f"Evaluating {task_name}"
    ):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        with torch.no_grad():
            outputs = model(**batch, head_mask=head_mask)

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


def collect_mask_grads(
    model: PreTrainedModel,
    dataloader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config = model.config

    # create masks
    head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).to(model.device, non_blocking=True)
    neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).to(model.device, non_blocking=True)
    ff_layer_mask = torch.ones(config.num_hidden_layers).to(model.device, non_blocking=True)
    attention_layer_mask = torch.ones(config.num_hidden_layers).to(model.device, non_blocking=True)
    hidden_state_mask = torch.ones(config.intermediate_size).to(model.device, non_blocking=True)

    # Requires grad to save it
    head_mask.requires_grad_(True)
    neuron_mask.requires_grad_(True)
    ff_layer_mask.requires_grad_(True)
    attention_layer_mask.requires_grad_(True)
    hidden_state_mask.requires_grad_(True)

    # TODO: apply masks to model
    # handles = apply_neuron_mask(model, neuron_mask)

    model.eval()
    head_grads, neuron_grads, ff_layer_grads, attention_layer_grads, hidden_state_grads = [], [], [], [], []
    for batch in tqdm(dataloader, total=len(dataloader), desc="Collecting grads"):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        outputs = model(head_mask=head_mask, **batch)
        loss = outputs.loss
        loss.backward()

        head_grads.append(head_mask.grad.detach())
        head_mask.grad = None

        # neuron_grads.append(neuron_mask.grad.detach())
        # neuron_mask.grad = None

    # TODO: remove masks from the model
    # for handle in handles:
    #     handle.remove()
    head_mask.requires_grad_(False)
    neuron_mask.requires_grad_(False)
    ff_layer_mask.requires_grad_(False)
    attention_layer_mask.requires_grad_(False)
    hidden_state_mask.requires_grad_(False)

    head_grads = torch.stack(head_grads, dim=0)
    # neuron_grads = torch.stack(neuron_grads, dim=0)
    # ff_layer_grads = torch.stack(ff_layer_grads, dim=0)
    # attention_layer_grads = torch.stack(attention_layer_grads, dim=0)
    # hidden_state_grads = torch.stack(hidden_state_grads, dim=0)
    return head_grads, None, None, None, None


def collect_activations(
    model: PreTrainedModel,
    dataloader: DataLoader,
) -> torch.Tensor:
    model.eval()
    # collect all head activations
    head_activations = []  # list[(batch_size, num_layers, num_heads)]
    for batch in tqdm(dataloader, total=len(dataloader), desc="Collecting activations"):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        outputs = model(**batch, output_attentions=True)
        attentions = outputs.attentions  # list of[(batch_size, num_heads, seq_len, seq_len)]
        # average over sequence length (last 2 dimensions)
        head_activations.append(
            torch.stack([
                layer.mean(dim=-1).mean(dim=-1)  # (batch_size, num_heads)
                for layer in attentions
            ], dim=1)
        )

    head_activations = torch.cat(head_activations, dim=0)  # (num_samples, num_layers, num_heads)
    return head_activations


@click.command()
@click.option("--model_name", type=str, default="bert-base-uncased")
@click.option("--task_name", type=str, default="mnli")
@click.option(
    "--pretrained_model_name_or_path",
    type=str,
    default="JeremiahZ/bert-base-uncased-mnli",
)
@click.option("--batch_size", type=int, default=32)
@click.option("--how_to_collect", type=str, default="grads")  # grads or activations or random
@click.option("--how_to_average", type=str, default="fisher_info")  # fisher_info or mean or entropy
@click.option("--num_samples", type=int, default=2048)
@click.option("--num_valid_samples", type=int, default=2048)
@click.option("--seed", type=int, default=0)
def main(
    model_name: str,
    task_name: str,
    pretrained_model_name_or_path: str,
    batch_size: int,
    how_to_collect: str,
    how_to_average: str,
    num_samples: int,
    num_valid_samples: int,
    seed: int,
) -> None:

    # Load the finetuned model and the corresponding tokenizer
    print(f"Loading model {pretrained_model_name_or_path}...")
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path, config=config
    )
    for param in model.parameters():
        param.requires_grad_(False)
    if IS_CUDA_AVAILABLE:
        model = model.to("cuda", non_blocking=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print("Model loaded")

    # load dataset
    print(f"Loading dataset {task_name}...")
    dataset = load_glue_dataset(task_name, tokenizer)
    collate_fn = DataCollatorWithPadding(tokenizer)
    print(dataset)

    sample_dataloader = DataLoader(
        dataset["train"].select(range(num_samples)) if len(dataset["train"]) > num_samples else dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"Sample dataloader: {len(sample_dataloader.dataset)} samples")

    validate_dataloader = DataLoader(
        dataset["validation"].select(range(num_valid_samples)) if len(dataset["validation"]) > num_valid_samples else dataset["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"Validation dataloader: {len(validate_dataloader.dataset)} samples")
    print("Dataset loaded")

    # measure before metric
    if IS_FP16_AVAILABLE:
        with torch.cuda.amp.autocast():
            metric, elapsed_time = measure_metric(
                model,
                None,
                validate_dataloader,
                task_name,
            )
    else:
        metric, elapsed_time = measure_metric(
            model,
            None,
            validate_dataloader,
            task_name,
        )
    print(f"BEFORE: {task_name} metric: {metric} {elapsed_time:.2f}s ({len(validate_dataloader.dataset)} samples)")

    if how_to_collect == "grads":
        # collect grads
        if IS_FP16_AVAILABLE:
            with torch.cuda.amp.autocast():
                head_grads, neuron_grads, ff_layer_grads, attention_layer_grads, hidden_state_grads = collect_mask_grads(
                    model,
                    sample_dataloader,
                )
        else:
            head_grads, neuron_grads, ff_layer_grads, attention_layer_grads, hidden_state_grads = collect_mask_grads(
                model,
                sample_dataloader,
            )
        print('head_grads', head_grads.shape)

        if how_to_average == "fisher_info":
            head_importance = head_grads.pow(2).sum(dim=0)
        elif how_to_average == "mean":
            head_importance = head_grads.abs().mean(dim=0)
        elif how_to_average == "entropy":
            # more entropy means more important - less predictable
            head_importance = - (head_grads * torch.log(head_grads)).sum(dim=0)
        else:
            assert False, f"Unknown how_to_average: {how_to_average}"
        print('head_importance', head_importance.shape)

    elif how_to_collect == "activations":
        # collect activations
        if IS_FP16_AVAILABLE:
            with torch.cuda.amp.autocast():
                head_activations = collect_activations(
                    model,
                    sample_dataloader,
                )
        else:
            head_activations = collect_activations(
                model,
                sample_dataloader,
            )
        print('head_activations', head_activations.shape)

        if how_to_average == "fisher_info":
            head_importance = head_activations.pow(2).sum(dim=0)
        elif how_to_average == "mean":
            head_importance = head_activations.abs().mean(dim=0)
        elif how_to_average == "entropy":
            # more entropy means more important - less predictable
            head_importance = - (head_activations * torch.log(head_activations)).sum(dim=0)
        else:
            assert False, f"Unknown how_to_average: {how_to_average}"
        print('head_importance', head_importance.shape)

    elif how_to_collect == "random":
        head_importance = torch.rand(config.num_hidden_layers, config.num_attention_heads)
        print('head_importance', head_importance.shape)

    else:
        assert False, f"Unknown how_to_collect: {how_to_collect}"

    # print average importance
    for layer in range(config.num_hidden_layers):
        print(f"> Layer {layer}: {head_importance[layer].mean().item()}")

    # prune
    for remain_head_percentage in [0.9, 0.7, 0.5, 0.3, 0.1]:
        # head_importance  # [num_hidden_layers, num_attention_heads]
        sorted_indices = head_importance.flatten().argsort(descending=True)
        cutoff_index = int(len(sorted_indices) * remain_head_percentage)
        head_mask = torch.ones_like(head_importance)
        head_mask.view(-1)[sorted_indices[cutoff_index:]] = 0
        head_mask = head_mask.to(model.device, non_blocking=True)

        # measure before metric
        if IS_FP16_AVAILABLE:
            with torch.cuda.amp.autocast():
                metric, elapsed_time = measure_metric(
                    model,
                    head_mask,  # [num_hidden_layers, num_attention_heads]
                    validate_dataloader,
                    task_name,
                )
        else:
            metric, elapsed_time = measure_metric(
                model,
                head_mask,  # [num_hidden_layers, num_attention_heads]
                validate_dataloader,
                task_name,
            )
        print(f"AFTER {remain_head_percentage*100}% remain on {how_to_collect}/{how_to_average}: {task_name} metric: {metric} {elapsed_time:.2f}s ({len(validate_dataloader.dataset)} samples)")

        # print layers and number of heads deleted
        for layer in range(config.num_hidden_layers):
            num_heads_deleted = config.num_attention_heads - head_mask[layer].sum().item()
            print(f"> Layer {layer}: {num_heads_deleted} heads deleted")
        print(f"percent of heads deleted: {1 - head_mask.sum().item() / head_mask.numel()}")


if __name__ == "__main__":
    main()
