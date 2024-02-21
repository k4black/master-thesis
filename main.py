from __future__ import annotations

import time

import click
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel, PreTrainedTokenizer,
)
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from evaluate import load
from tqdm import tqdm
from torch.utils.hooks import RemovableHandle
import pandas as pd

from adaptive_pruning.pruning import (
    prune_attention_heads, prune_attention_layers, prune_ffn_neurons, prune_ffn_layers, do_prune_hidden_state,
    select_to_prune_attention_heads, select_to_prune_attention_layers, select_to_prune_ffn_neurons,
    select_to_prune_ffn_layers
)
from adaptive_pruning.injections import (
    inject_attention_head_mask, inject_attention_layer_mask, inject_ffn_neuron_mask, inject_ffn_layer_mask,
)
from adaptive_pruning.utils import count_parameters


CUDA_DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
IS_TPU_AVAILABLE = torch.cuda.is_available() and "TPU" in torch.cuda.get_device_name(0)
IS_CUDA_AVAILABLE = torch.cuda.is_available() and not IS_TPU_AVAILABLE
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE
# IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.backends.cuda.is_bf16_supported
print(f"CUDA_DEVICES: {CUDA_DEVICES}")
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")
print(f"FP16_AVAILABLE: {IS_FP16_AVAILABLE}")
# print(f"BF16_AVAILABLE: {IS_BF16_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    def _tokenize(examples: dict) -> dict:
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
    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Evaluating {task_name}", leave=False):
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


def collect_mask_grads(
    model: PreTrainedModel,
    dataloader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config = model.config

    # create masks
    head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).to(model.device, non_blocking=True)
    ff_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).to(model.device, non_blocking=True)
    ff_layer_mask = torch.ones(config.num_hidden_layers).to(model.device, non_blocking=True)
    attention_layer_mask = torch.ones(config.num_hidden_layers).to(model.device, non_blocking=True)
    hidden_state_mask = torch.ones(config.hidden_size).to(model.device, non_blocking=True)

    # Requires grad to save it
    head_mask.requires_grad_(True)
    ff_neuron_mask.requires_grad_(True)
    ff_layer_mask.requires_grad_(True)
    attention_layer_mask.requires_grad_(True)
    hidden_state_mask.requires_grad_(True)

    # apply masks to model
    handles: list[RemovableHandle] = [
        *inject_attention_head_mask(model.bert, head_mask),
        *inject_attention_layer_mask(model.bert, attention_layer_mask),
        *inject_ffn_neuron_mask(model.bert, ff_neuron_mask),
        *inject_ffn_layer_mask(model.bert, ff_layer_mask),
        # *inject_hidden_state_mask(model.bert, hidden_state_mask),
    ]

    model.eval()
    head_grads, ff_neuron_grads, ff_layer_grads, attention_layer_grads, hidden_state_grads = [], [], [], [], []
    for batch in tqdm(dataloader, total=len(dataloader), desc="Collecting grads"):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        # outputs = model(head_mask=head_mask, **batch)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        head_grads.append(head_mask.grad.detach())
        head_mask.grad = None

        ff_neuron_grads.append(ff_neuron_mask.grad.detach())
        ff_neuron_mask.grad = None

        ff_layer_grads.append(ff_layer_mask.grad.detach())
        ff_layer_mask.grad = None

        attention_layer_grads.append(attention_layer_mask.grad.detach())
        attention_layer_mask.grad = None

        # hidden_state_grads.append(hidden_state_mask.grad.detach())
        # hidden_state_mask.grad = None

    # remove masks from the model
    for handle in handles:
        handle.remove()

    # disable grad
    head_mask.requires_grad_(False)
    ff_neuron_mask.requires_grad_(False)
    ff_layer_mask.requires_grad_(False)
    attention_layer_mask.requires_grad_(False)
    hidden_state_mask.requires_grad_(False)

    head_grads = torch.stack(head_grads, dim=0)
    ff_neuron_grads = torch.stack(ff_neuron_grads, dim=0)
    ff_layer_grads = torch.stack(ff_layer_grads, dim=0)
    attention_layer_grads = torch.stack(attention_layer_grads, dim=0)
    # hidden_state_grads = torch.stack(hidden_state_grads, dim=0)
    return head_grads, ff_neuron_grads, ff_layer_grads, attention_layer_grads, None


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
            torch.stack([layer.mean(dim=-1).mean(dim=-1) for layer in attentions], dim=1)  # (batch_size, num_heads)
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
@click.option("--num_samples", type=int, default=256)
@click.option("--num_valid_samples", type=int, default=256)
@click.option("--do_prune_attention_heads", is_flag=True, default=False)
@click.option("--do_prune_attention_layers", is_flag=True, default=False)
@click.option("--do_prune_ffn_neurons", is_flag=True, default=False)
@click.option("--do_prune_ffn_layers", is_flag=True, default=False)
@click.option("--do_prune_hidden_state", is_flag=True, default=False)
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
    do_prune_attention_heads: bool,
    do_prune_attention_layers: bool,
    do_prune_ffn_neurons: bool,
    do_prune_ffn_layers: bool,
    do_prune_hidden_state: bool,
    seed: int,
) -> None:
    # Load the finetuned model and the corresponding tokenizer
    print(f"Loading model {pretrained_model_name_or_path}...")
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config)
    if IS_CUDA_AVAILABLE:
        model = model.to("cuda", non_blocking=True)
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(f"Number of parameters: {count_parameters(model)}")
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
        (
            dataset["validation"].select(range(num_valid_samples))
            if len(dataset["validation"]) > num_valid_samples
            else dataset["validation"]
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"Validation dataloader: {len(validate_dataloader.dataset)} samples")
    print("Dataset loaded")

    # print model with sample input
    # summary(model, input_size=(batch_size, 512), dtypes=['torch.IntTensor'], depth=7, device=model.device)
    print(model)
    print(model.config)

    print("-" * 80)

    # measure before metric
    # metric, elapsed_time = measure_glue_metric(
    #     model,
    #     validate_dataloader,
    #     task_name,
    # )
    # print(f"BEFORE: {task_name} metric: {metric:.4f} {elapsed_time:.2f}s ({len(validate_dataloader.dataset)} samples)")

    if how_to_collect == "grads":
        # collect grads
        attention_head_grads, ff_neuron_grads, ff_layer_grads, attention_layer_grads, hidden_state_grads = (
            collect_mask_grads(
                model,
                sample_dataloader,
            )
        )

        if how_to_average == "fisher_info":
            # Note: actually mean in formula, but for accurate calculation - can just take sum
            attention_head_importance = attention_head_grads.pow(2).sum(dim=0)
            ff_neuron_importance = ff_neuron_grads.pow(2).sum(dim=0)
            ff_layer_importance = ff_layer_grads.pow(2).sum(dim=0)
            attention_layer_importance = attention_layer_grads.pow(2).sum(dim=0)
            # hidden_state_importance = hidden_state_grads.pow(2).sum(dim=0)
        elif how_to_average == "mean":
            attention_head_importance = attention_head_grads.abs().mean(dim=0)
            ff_neuron_importance = ff_neuron_grads.abs().mean(dim=0)
            ff_layer_importance = ff_layer_grads.abs().mean(dim=0)
            attention_layer_importance = attention_layer_grads.abs().mean(dim=0)
            # hidden_state_importance = hidden_state_grads.abs().mean(dim=0)
        elif how_to_average == "entropy":
            # more entropy means more important - less predictable
            attention_head_importance = -(attention_head_grads * torch.log(attention_head_grads)).sum(dim=0)
            ff_neuron_importance = -(ff_neuron_grads * torch.log(ff_neuron_grads)).sum(dim=0)
            ff_layer_importance = -(ff_layer_grads * torch.log(ff_layer_grads)).sum(dim=0)
            attention_layer_importance = -(attention_layer_grads * torch.log(attention_layer_grads)).sum(dim=0)
            # hidden_state_importance = - (hidden_state_grads * torch.log(hidden_state_grads)).sum(dim=0)
        else:
            assert False, f"Unknown how_to_average: {how_to_average}"
        print("attention_head_importance", attention_head_importance.shape)

    elif how_to_collect == "activations":
        raise NotImplementedError("Now only implemented for attention heads, wait")

        # collect activations
        head_activations = collect_activations(
            model,
            sample_dataloader,
        )

        if how_to_average == "fisher_info":
            attention_head_importance = head_activations.pow(2).sum(dim=0)
        elif how_to_average == "mean":
            attention_head_importance = head_activations.abs().mean(dim=0)
        elif how_to_average == "entropy":
            # more entropy means more important - less predictable
            attention_head_importance = -(head_activations * torch.log(head_activations)).sum(dim=0)
        else:
            assert False, f"Unknown how_to_average: {how_to_average}"
        print("attention_head_importance", attention_head_importance.shape)

    elif how_to_collect == "random":
        attention_head_importance = torch.rand(config.num_hidden_layers, config.num_attention_heads)
        ff_neuron_importance = torch.rand(config.num_hidden_layers, config.intermediate_size)
        ff_layer_importance = torch.rand(config.num_hidden_layers)
        attention_layer_importance = torch.rand(config.num_hidden_layers)
        hidden_state_importance = torch.rand(config.hidden_size)
        print("attention_head_importance", attention_head_importance.shape)

    else:
        assert False, f"Unknown how_to_collect: {how_to_collect}"

    # print average importance
    print(f"Average importance for {how_to_collect}/{how_to_average}:")
    for layer in range(config.num_hidden_layers):
        print(f"> Layer {layer}:")
        print(f"  Attention Head: {attention_head_importance[layer].mean().item()}")
        print(f"  FF Neuron: {ff_neuron_importance[layer].mean().item()}")
        print(f"  FF Layer: {ff_layer_importance[layer].mean().item()}")
        print(f"  Attention Layer: {attention_layer_importance[layer].mean().item()}")
        # print(f"  Hidden State: {hidden_state_importance.mean().item()}")

    # prune
    df_stats = pd.DataFrame(columns=["remain_percentage", "metric", "elapsed_time", "params_num"])
    for remain_percentage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        print("-" * 80)
        print(
            f"Pruning {(1 - remain_percentage) * 100:.0f}% with {remain_percentage * 100:.0f}% remain "
            f"on {how_to_collect}/{how_to_average}..."
        )

        # load fresh model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, config=config, ignore_mismatched_sizes=True
        )
        for param in model.parameters():
            param.requires_grad_(False)

        attention_layers_to_prune = []
        if do_prune_attention_layers:
            # convert importance to layers to prune
            attention_layers_to_prune = select_to_prune_attention_layers(attention_layer_importance, 1-remain_percentage)
            # actually prune layers
            prune_attention_layers(model.bert, attention_layers_to_prune)
            # print layers deleted
            total_percent_layers_deleted = len(attention_layers_to_prune) / model.config.num_hidden_layers
            print(
                f"Total: {total_percent_layers_deleted * 100:.2f}% attention layers deleted, "
                f"{(1 - total_percent_layers_deleted) * 100:.2f}% remain"
            )

        if do_prune_attention_heads:
            # convert importance to heads to prune (skip pruned attention layers)
            attention_heads_to_prune = select_to_prune_attention_heads(attention_head_importance, 1-remain_percentage, uniform_among_layers=True)
            attention_heads_to_prune = {
                layer: heads for layer, heads in attention_heads_to_prune.items() if layer not in attention_layers_to_prune
            }
            # actually prune heads
            prune_attention_heads(model.bert, attention_heads_to_prune)
            # print layers and number of heads deleted
            total_num_heads_deleted = 0
            for layer in range(config.num_hidden_layers):
                num_heads_deleted = (
                    config.num_attention_heads - model.bert.encoder.layer[layer].attention.self.num_attention_heads
                )
                total_num_heads_deleted += num_heads_deleted
                print(f"> Layer {layer}: {num_heads_deleted} attention heads deleted")
            total_percent_heads_deleted = total_num_heads_deleted / (
                config.num_attention_heads * config.num_hidden_layers
            )
            print(
                f"Total: {total_percent_heads_deleted * 100:.2f}% attention heads deleted, "
                f"{(1 - total_percent_heads_deleted) * 100:.2f}% remain"
            )

        if do_prune_ffn_layers:
            # convert importance to layers to prune
            ffn_layers_to_prune = select_to_prune_ffn_layers(ff_layer_importance, 1-remain_percentage)
            # actually prune layers
            prune_ffn_layers(model.bert, ffn_layers_to_prune)
            # print layers deleted
            total_percent_layers_deleted = len(ffn_layers_to_prune) / model.config.num_hidden_layers
            print(
                f"Total: {total_percent_layers_deleted * 100:.2f}% ffn layers deleted, "
                f"{(1 - total_percent_layers_deleted) * 100:.2f}% remain"
            )

        if do_prune_ffn_neurons:
            # convert importance to neurons to prune (skip pruned ffn layers)
            neurons_to_prune = select_to_prune_ffn_neurons(ff_neuron_importance, 1-remain_percentage, uniform_among_layers=True)
            neurons_to_prune = {
                layer: neurons for layer, neurons in neurons_to_prune.items() if layer not in attention_layers_to_prune
            }
            # actually prune neurons
            prune_ffn_neurons(model.bert, neurons_to_prune)
            # print layers and number of neurons deleted in FF
            total_num_neurons_deleted = 0
            for layer in range(config.num_hidden_layers):
                num_neurons_deleted = (
                    config.intermediate_size - model.bert.encoder.layer[layer].intermediate.dense.out_features
                )
                total_num_neurons_deleted += num_neurons_deleted
                print(f"> Layer {layer}: {num_neurons_deleted} neurons deleted")
            total_percent_neurons_deleted = total_num_neurons_deleted / (
                config.intermediate_size * config.num_hidden_layers
            )
            print(
                f"Total: {total_percent_neurons_deleted * 100:.2f}% neurons deleted, "
                f"{(1 - total_percent_neurons_deleted) * 100:.2f}% remain"
            )

        if IS_CUDA_AVAILABLE:
            model = model.to("cuda", non_blocking=True)

        # measure before metric
        metric, elapsed_time = measure_glue_metric(
            model,
            validate_dataloader,
            task_name,
        )
        df_stats.loc[len(df_stats)] = {
            "remain_percentage": remain_percentage,
            "metric": round(metric, 4),
            "elapsed_time": round(elapsed_time, 2),
            "params_num": count_parameters(model),
        }
        print(
            f"AFTER {remain_percentage * 100}% remain on {how_to_collect}/{how_to_average}: "
            f"{task_name} metric: {metric:.4f} {elapsed_time:.2f}s ({len(validate_dataloader.dataset)} samples)"
        )

    # print stats
    # get 1.0 remain_percentage row
    max_metric_row = df_stats[df_stats["remain_percentage"] == 1.0].iloc[0]
    df_stats['relative_metric'] = round(df_stats['metric'] / max_metric_row['metric'], 2)
    df_stats['relative_elapsed_time'] = round(df_stats['elapsed_time'] / max_metric_row['elapsed_time'], 2)
    df_stats['relative_params_num'] = round(df_stats['params_num'] / max_metric_row['params_num'], 2)
    print("+" * 80)
    with pd.option_context(
            'display.max_rows',
            None,
            'display.max_columns',
            None,
            'display.expand_frame_repr',
            False,
            'max_colwidth',
            -1
    ):
        print(df_stats)


if __name__ == "__main__":
    if IS_FP16_AVAILABLE:
        with torch.cuda.amp.autocast():
            main()
    else:
        main()
