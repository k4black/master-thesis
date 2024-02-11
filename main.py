from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Generator, Callable

import click
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
)
from transformers.pytorch_utils import prune_linear_layer
from tokenizers import Tokenizer
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from evaluate import load
from tqdm import tqdm
from torchinfo import summary
from torch.utils.hooks import RemovableHandle


IS_TPU_AVAILABLE = torch.cuda.is_available() and "TPU" in torch.cuda.get_device_name(0)
IS_CUDA_AVAILABLE = torch.cuda.is_available() and not IS_TPU_AVAILABLE
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE
# IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.backends.cuda.is_bf16_supported
print(f"all devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}")
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


def apply_ff_neuron_mask(model: PreTrainedModel, mask: torch.Tensor) -> Generator[RemovableHandle, None, None]:
    """
    Apply mask to FF neurons
    For each layer add masking in forward hook between 'intermediate' and 'output' layers
    We mask the input of the "output" layer so it is the same as delete the neurons of the prev (intermediate) layer

    -> [batch_size, seq_len, hidden_size]
    intermediate [hidden_size, intermediate_size]
    -> [batch_size, seq_len, intermediate_size]
    mask [intermediate_size]
    -> [batch_size, seq_len, intermediate_size]
    output [intermediate_size, hidden_size]
    -> [batch_size, seq_len, hidden_size]

    :param model: Torch model we can select layers from and add hooks to
    :param mask: Mask of [num_hidden_layers, intermediate_size]
    :return: yield of hooks to remove after use
    """

    for layer in range(model.config.num_hidden_layers):
        output_layer = model.bert.encoder.layer[layer].output
        mask_for_layer = mask[layer]
        mask_hook = lambda module, inputs: (inputs[0] * mask_for_layer, inputs[1])
        output_handle = output_layer.register_forward_pre_hook(mask_hook)
        yield output_handle


def apply_ff_layer_mask(model: PreTrainedModel, mask: torch.Tensor) -> Generator[RemovableHandle, None, None]:
    """
    Apply mask to the whole FF layers
    For each layer add masking in forward hook between 'intermediate' and 'output' layers
    We mask the whole input of the "output" layer so it is the same as delete the whole FF layer

    -> [batch_size, seq_len, hidden_size]
    intermediate [hidden_size, intermediate_size]
    -> [batch_size, seq_len, intermediate_size]
    mask [1]
    -> [batch_size, seq_len, intermediate_size]
    output [intermediate_size, hidden_size]
    -> [batch_size, seq_len, hidden_size]

    :param model: Torch model we can select layers from and add hooks to
    :param mask: Mask of [num_hidden_layers]
    :return: yield of hooks to remove after use
    """

    for layer in range(model.config.num_hidden_layers):
        output_layer = model.bert.encoder.layer[layer].output
        mask_for_layer = mask[layer]
        mask_hook = lambda module, inputs: (inputs[0] * mask_for_layer, inputs[1])
        output_handle = output_layer.register_forward_pre_hook(mask_hook)
        yield output_handle


def apply_attention_layer_mask(model: PreTrainedModel, mask: torch.Tensor) -> Generator[RemovableHandle, None, None]:
    """
    Apply mask to the whole attention layer
    For each layer add masking in forward hook between 'self' and 'output' layers
    We mask the whole input of the "output" layer so it is the same as delete the whole attention layer

    -> [batch_size, seq_len, hidden_size]
    self [hidden_size, hidden_size]
    -> [batch_size, seq_len, hidden_size]
    mask [1]
    -> [batch_size, seq_len, hidden_size]
    output [hidden_size, hidden_size]
    -> [batch_size, seq_len, hidden_size]

    :param model: Torch model we can select layers from and add hooks to
    :param mask: Mask of [num_hidden_layers]
    :return: yield of hooks to remove after use
    """

    for layer in range(model.config.num_hidden_layers):
        output_layer = model.bert.encoder.layer[layer].attention.output
        mask_for_layer = mask[layer]
        mask_hook = lambda module, inputs: (inputs[0] * mask_for_layer, inputs[1])
        output_handle = output_layer.register_forward_pre_hook(mask_hook)
        yield output_handle


def apply_hidden_state_mask(model: PreTrainedModel, mask: torch.Tensor) -> Generator[RemovableHandle, None, None]:
    """
    Apply mask to neurons of the hidden state
    For each layer/weight add masking in forward hook to mask some hidden state neurons
    We apply the same mask on each layer as we have residual connections

    -> [batch_size, seq_len, hidden_size]
    mask [hidden_size]
    -> [batch_size, seq_len, hidden_size]
    some_weight [hidden_size, hidden_size]
    -> [batch_size, seq_len, hidden_size]

    :param model: Torch model we can select layers from and add hooks to
    :param mask: Mask of [hidden_size]
    :return: yield of hooks to remove after use
    """
    raise NotImplementedError("Not implemented yet")


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
    handles: list[RemovableHandle] = []
    handles.extend(
        apply_ff_neuron_mask(model, ff_neuron_mask)
    )
    handles.extend(
        apply_ff_layer_mask(model, ff_layer_mask)
    )
    handles.extend(
        apply_attention_layer_mask(model, attention_layer_mask)
    )
    # handles.extend(
    #     apply_hidden_state_mask(model, hidden_state_mask)
    # )

    model.eval()
    head_grads, ff_neuron_grads, ff_layer_grads, attention_layer_grads, hidden_state_grads = [], [], [], [], []
    for batch in tqdm(dataloader, total=len(dataloader), desc="Collecting grads"):
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        outputs = model(head_mask=head_mask, **batch)
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
@click.option("--num_samples", type=int, default=256)
@click.option("--num_valid_samples", type=int, default=256)
@click.option("--prune_heads", is_flag=True, default=False)
@click.option("--prune_ff_neurons", is_flag=True, default=False)
@click.option("--prune_hidden_state", is_flag=True, default=False)
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
    prune_heads: bool,
    prune_ff_neurons: bool,
    prune_hidden_state: bool,
    seed: int,
) -> None:

    # Load the finetuned model and the corresponding tokenizer
    print(f"Loading model {pretrained_model_name_or_path}...")
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path, config=config
    )
    if IS_CUDA_AVAILABLE:
        model = model.to("cuda", non_blocking=True)
    for param in model.parameters():
        param.requires_grad_(False)
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

    # print model with sample input
    # summary(model, input_size=(batch_size, 512), dtypes=['torch.IntTensor'], depth=7, device=model.device)
    print(model)
    print(model.config)

    print('-' * 80)

    # measure before metric
    if IS_FP16_AVAILABLE:
        with torch.cuda.amp.autocast():
            metric, elapsed_time = measure_metric(
                model,
                validate_dataloader,
                task_name,
            )
    else:
        metric, elapsed_time = measure_metric(
            model,
            validate_dataloader,
            task_name,
        )
    print(f"BEFORE: {task_name} metric: {metric:.4f} {elapsed_time:.2f}s ({len(validate_dataloader.dataset)} samples)")

    if how_to_collect == "grads":
        # collect grads
        if IS_FP16_AVAILABLE:
            with torch.cuda.amp.autocast():
                head_grads, ff_neuron_grads, ff_layer_grads, attention_layer_grads, hidden_state_grads = collect_mask_grads(
                    model,
                    sample_dataloader,
                )
        else:
            head_grads, ff_neuron_grads, ff_layer_grads, attention_layer_grads, hidden_state_grads = collect_mask_grads(
                model,
                sample_dataloader,
            )

        if how_to_average == "fisher_info":
            head_importance = head_grads.pow(2).sum(dim=0)
            ff_neuron_importance = ff_neuron_grads.pow(2).sum(dim=0)
            ff_layer_importance = ff_layer_grads.pow(2).sum(dim=0)
            attention_layer_importance = attention_layer_grads.pow(2).sum(dim=0)
            # hidden_state_importance = hidden_state_grads.pow(2).sum(dim=0)
        elif how_to_average == "mean":
            head_importance = head_grads.abs().mean(dim=0)
            ff_neuron_importance = ff_neuron_grads.abs().mean(dim=0)
            ff_layer_importance = ff_layer_grads.abs().mean(dim=0)
            attention_layer_importance = attention_layer_grads.abs().mean(dim=0)
            # hidden_state_importance = hidden_state_grads.abs().mean(dim=0)
        elif how_to_average == "entropy":
            # more entropy means more important - less predictable
            head_importance = - (head_grads * torch.log(head_grads)).sum(dim=0)
            ff_neuron_importance = - (ff_neuron_grads * torch.log(ff_neuron_grads)).sum(dim=0)
            ff_layer_importance = - (ff_layer_grads * torch.log(ff_layer_grads)).sum(dim=0)
            attention_layer_importance = - (attention_layer_grads * torch.log(attention_layer_grads)).sum(dim=0)
            # hidden_state_importance = - (hidden_state_grads * torch.log(hidden_state_grads)).sum(dim=0)
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
        ff_neuron_importance = torch.rand(config.num_hidden_layers, config.intermediate_size)
        ff_layer_importance = torch.rand(config.num_hidden_layers)
        attention_layer_importance = torch.rand(config.num_hidden_layers)
        hidden_state_importance = torch.rand(config.hidden_size)
        print('head_importance', head_importance.shape)

    else:
        assert False, f"Unknown how_to_collect: {how_to_collect}"

    # print average importance
    print(f"Average importance for {how_to_collect}/{how_to_average}:")
    for layer in range(config.num_hidden_layers):
        print(f"> Layer {layer}:")
        print(f"  Attention Head: {head_importance[layer].mean().item()}")
        print(f"  FF Neuron: {ff_neuron_importance[layer].mean().item()}")
        print(f"  FF Layer: {ff_layer_importance[layer].mean().item()}")
        print(f"  Attention Layer: {attention_layer_importance[layer].mean().item()}")
        # print(f"  Hidden State: {hidden_state_importance.mean().item()}")

    # prune
    for remain_head_percentage in [0.9, 0.7, 0.5, 0.3, 0.1]:
        print('-'*80)
        print(f"Pruning {remain_head_percentage*100}% remain on {how_to_collect}/{how_to_average}...")

        # load fresh model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, config=config, ignore_mismatched_sizes=True  #  TODO: check why this is needed
        )
        for param in model.parameters():
            param.requires_grad_(False)
        if IS_CUDA_AVAILABLE:
            model = model.to("cuda", non_blocking=True)

        if prune_heads:
            # head_importance  # [num_hidden_layers, num_attention_heads]
            # collect heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            # note: first select top k% heads to prune regardless of layer
            heads_importance_flat = head_importance.view(-1)
            num_heads_to_prune = int(head_importance.numel() * (1 - remain_head_percentage))
            _, heads_to_prune_flat = heads_importance_flat.topk(num_heads_to_prune, largest=False)
            heads_to_prune = defaultdict(list)
            for head_idx in heads_to_prune_flat:
                layer_idx = head_idx // config.num_attention_heads
                head_idx = head_idx % config.num_attention_heads
                heads_to_prune[layer_idx.item()].append(head_idx)
            # keep at least 1 head per layer
            for layer in range(config.num_hidden_layers):
                if len(heads_to_prune[layer]) == config.num_attention_heads:
                    heads_to_prune[layer].pop()

            # actually prune heads
            model.prune_heads(heads_to_prune)

            # print layers and number of heads deleted
            total_num_heads_deleted = 0
            for layer in range(config.num_hidden_layers):
                num_heads_deleted = config.num_attention_heads - model.bert.encoder.layer[layer].attention.self.num_attention_heads
                total_num_heads_deleted += num_heads_deleted
                print(f"> Layer {layer}: {num_heads_deleted} heads deleted")
            total_percent_heads_deleted = total_num_heads_deleted / (config.num_attention_heads * config.num_hidden_layers)
            print(f"Total: {total_percent_heads_deleted*100:.2f}% heads deleted, {(1-total_percent_heads_deleted)*100:.2f}% remain")

        if prune_ff_neurons:
            # prune FF neurons
            # prune intermediate_size in 'intermediate' and 'output' layers
            # we prune the same neurons in both layers
            for layer in range(config.num_hidden_layers):
                ff_intermediate = model.bert.encoder.layer[layer].intermediate.dense
                ff_output = model.bert.encoder.layer[layer].output.dense
                neurons_to_prune = ff_neuron_importance[layer].argsort()[:int(config.intermediate_size * remain_head_percentage)]
                model.bert.encoder.layer[layer].intermediate.dense = prune_linear_layer(ff_intermediate, neurons_to_prune, dim=0)
                model.bert.encoder.layer[layer].output.dense = prune_linear_layer(ff_output, neurons_to_prune, dim=1)

            # print layers and number of neurons deleted in FF
            total_num_neurons_deleted = 0
            for layer in range(config.num_hidden_layers):
                num_neurons_deleted = config.intermediate_size - model.bert.encoder.layer[layer].intermediate.dense.out_features
                total_num_neurons_deleted += num_neurons_deleted
                print(f"> Layer {layer}: {num_neurons_deleted} neurons deleted")
            total_percent_neurons_deleted = total_num_neurons_deleted / (config.intermediate_size * config.num_hidden_layers)
            print(f"Total: {total_percent_neurons_deleted*100:.2f}% neurons deleted, {(1-total_percent_neurons_deleted)*100:.2f}% remain")

        # measure before metric
        if IS_FP16_AVAILABLE:
            with torch.cuda.amp.autocast():
                metric, elapsed_time = measure_metric(
                    model,
                    validate_dataloader,
                    task_name,
                )
        else:
            metric, elapsed_time = measure_metric(
                model,
                validate_dataloader,
                task_name,
            )
        print(f"AFTER {remain_head_percentage*100}% remain on {how_to_collect}/{how_to_average}: {task_name} metric: {metric:.4f} {elapsed_time:.2f}s ({len(validate_dataloader.dataset)} samples)")


if __name__ == "__main__":
    main()
