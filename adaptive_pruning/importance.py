from __future__ import annotations

from collections.abc import Sized
from typing import NamedTuple, cast

import torch
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm
from transformers import PreTrainedModel

from adaptive_pruning.injections import (
    inject_attention_head_mask,
    inject_attention_layer_mask,
    inject_ffn_neuron_mask,
    inject_ffn_layer_mask,
    inject_hidden_state_mask,
)


class ComponentsInfo(NamedTuple):
    attention_heads_info: torch.Tensor  # [num_samples, num_layers, num_heads]
    attention_layers_info: torch.Tensor  # [num_samples, num_layers]
    ffn_neurons_info: torch.Tensor  # [num_samples, num_layers, hidden_state]
    ffn_layers_info: torch.Tensor  # [num_samples, num_layers]
    hidden_states_info: torch.Tensor  # [num_samples, hidden_state]
    meta_info: torch.Tensor  # [num_samples, 5] - attn_heads, attn_layers, ffn_neurons, ffn_layers, hidden_state


class ComponentsImportance(NamedTuple):
    attention_heads_importance: torch.Tensor  # [num_layers, num_heads]
    attention_layers_importance: torch.Tensor  # [num_layers]
    ffn_neurons_importance: torch.Tensor  # [num_layers, hidden_state]
    ffn_layers_importance: torch.Tensor  # [num_layers]
    hidden_states_importance: torch.Tensor  # [hidden_state]
    meta_importance: torch.Tensor  # [5] - attn_heads, attn_layers, ffn_neurons, ffn_layers, hidden_state


def collect_mask_grads(
        model: PreTrainedModel,
        dataloader: DataLoader,
        *,
        verbose: bool = True,
) -> ComponentsInfo:
    batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Collecting grads") if verbose else dataloader

    config = model.config

    # Disable grads for other model
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    # create masks
    attention_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).to(model.device,
                                                                                              non_blocking=True)
    ffn_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).to(model.device, non_blocking=True)
    ffn_layer_mask = torch.ones(config.num_hidden_layers).to(model.device, non_blocking=True)
    attention_layer_mask = torch.ones(config.num_hidden_layers).to(model.device, non_blocking=True)
    hidden_state_mask = torch.ones(config.hidden_size).to(model.device, non_blocking=True)

    # Requires grad to save it
    attention_head_mask.requires_grad_(True)
    ffn_neuron_mask.requires_grad_(True)
    ffn_layer_mask.requires_grad_(True)
    attention_layer_mask.requires_grad_(True)
    hidden_state_mask.requires_grad_(True)

    # Make meta-mask over all masks - 1 for each other mask
    meta_mask = torch.ones(5).to(model.device, non_blocking=True)
    meta_mask.requires_grad_(True)

    # apply masks to model
    handles: list[RemovableHandle] = [
        *inject_attention_head_mask(model.bert, attention_head_mask, meta_mask[0]),
        *inject_attention_layer_mask(model.bert, attention_layer_mask, meta_mask[1]),
        *inject_ffn_neuron_mask(model.bert, ffn_neuron_mask, meta_mask[2]),
        *inject_ffn_layer_mask(model.bert, ffn_layer_mask, meta_mask[3]),
        # *inject_hidden_state_mask(model_to_insert, hidden_state_mask, meta_mask[4]),
    ]

    attention_head_grads = torch.empty((0, config.num_hidden_layers, config.num_attention_heads)).to(model.device, non_blocking=True)
    attention_layer_grads = torch.empty((0, config.num_hidden_layers)).to(model.device, non_blocking=True)
    ffn_neuron_grads = torch.empty((0, config.num_hidden_layers, config.intermediate_size)).to(model.device, non_blocking=True)
    ffn_layer_grads = torch.empty((0, config.num_hidden_layers)).to(model.device, non_blocking=True)
    hidden_state_grads = torch.empty((0, config.hidden_size)).to(model.device, non_blocking=True)
    meta_grads = torch.empty((0, 5)).to(model.device, non_blocking=True)

    for batch in batch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        attention_head_grads = torch.cat([attention_head_grads, attention_head_mask.grad.detach().unsqueeze(0)], dim=0)
        attention_head_mask.grad = None

        attention_layer_grads = torch.cat([attention_layer_grads, attention_layer_mask.grad.detach().unsqueeze(0)],
                                          dim=0)
        attention_layer_mask.grad = None

        ffn_neuron_grads = torch.cat([ffn_neuron_grads, ffn_neuron_mask.grad.detach().unsqueeze(0)], dim=0)
        ffn_neuron_mask.grad = None

        ffn_layer_grads = torch.cat([ffn_layer_grads, ffn_layer_mask.grad.detach().unsqueeze(0)], dim=0)
        ffn_layer_mask.grad = None

        # hidden_state_grads = torch.cat([hidden_state_grads, hidden_state_mask.grad.detach().unsqueeze(0)], dim=0)
        hidden_state_mask.grad = None

        meta_grads = torch.cat([meta_grads, meta_mask.grad.detach().unsqueeze(0)], dim=0)
        meta_mask.grad = None

    # remove masks from the model
    for handle in handles:
        handle.remove()

    # disable grad
    attention_head_mask.requires_grad_(False)
    ffn_neuron_mask.requires_grad_(False)
    ffn_layer_mask.requires_grad_(False)
    attention_layer_mask.requires_grad_(False)
    hidden_state_mask.requires_grad_(False)

    return ComponentsInfo(
        attention_head_grads, attention_layer_grads, ffn_neuron_grads, ffn_layer_grads, hidden_state_grads, meta_grads,
    )


def collect_activations(
        model: PreTrainedModel,
        dataloader: DataLoader,
        *,
        verbose: bool = False,
) -> ComponentsInfo:
    batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Collecting activations") if verbose else dataloader

    config = model.config

    # Disable grads for other model
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    # Collect activations
    attention_head_activations = torch.empty((0, config.num_hidden_layers, config.num_attention_heads)).to(model.device, non_blocking=True)
    attention_layer_activations = torch.empty((0, config.num_hidden_layers)).to(model.device, non_blocking=True)
    ffn_neuron_activations = torch.empty((0, config.num_hidden_layers, config.intermediate_size)).to(model.device, non_blocking=True)
    ffn_layer_activations = torch.empty((0, config.num_hidden_layers)).to(model.device, non_blocking=True)
    hidden_state_activations = torch.empty((0, config.hidden_size)).to(model.device, non_blocking=True)
    meta_activations = torch.empty((0, 5)).to(model.device, non_blocking=True)
    for batch in batch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        outputs = model(**batch, output_attentions=True, output_hidden_states=True)

        # list by layers of[(batch_size, num_heads, seq_len, seq_len)]
        # average over sequence length (last 2 dimensions) - to get (batch_size, num_layers, num_heads)
        stacked_attention = torch.stack([layer.mean(dim=-1).mean(dim=-1) for layer in outputs.attentions], dim=1)
        attention_head_activations = torch.cat([attention_head_activations, stacked_attention], dim=0)
        # average over attention heads
        attention_layer_activations = torch.cat([attention_layer_activations, stacked_attention.mean(dim=-1)], dim=0)

        # TODO: extract real neurons activations

        # list by layers of [(batch_size, seq_len, hidden_states)] - note - first one if embeddings output
        hidden_stated_with_embeddings = [layer.mean(dim=1) for layer in outputs.hidden_states]
        hidden_state_activations = torch.cat(
            [hidden_state_activations, torch.stack(hidden_stated_with_embeddings, dim=1).mean(dim=1)], dim=0)

    return ComponentsInfo(
        attention_head_activations,
        attention_layer_activations,
        torch.Tensor(),
        torch.Tensor(),
        hidden_state_activations,
        torch.Tensor(),
    )


def collect_weight_magnitudes(
        model: PreTrainedModel,
        dataloader: DataLoader | None = None,  # for testing
) -> ComponentsInfo:
    config = model.config

    # Note: in case of weights we will have 1 sample (dim=0)
    attention_head_magnitudes = torch.empty((1, config.num_hidden_layers, config.num_attention_heads)).to(model.device, non_blocking=True)
    attention_layer_magnitudes = torch.empty((1, config.num_hidden_layers)).to(model.device, non_blocking=True)
    ffn_neuron_magnitudes = torch.empty((1, config.num_hidden_layers, config.intermediate_size)).to(model.device, non_blocking=True)
    ffn_layer_magnitudes = torch.empty((1, config.num_hidden_layers)).to(model.device, non_blocking=True)
    hidden_state_magnitudes = torch.empty((1, config.hidden_size)).to(model.device, non_blocking=True)
    meta_magnitudes = torch.empty((1, 5)).to(model.device, non_blocking=True)

    pass

    return ComponentsInfo(
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
        torch.Tensor(),
    )


def collect_random_numbers(
        model: PreTrainedModel,
        dataloader: DataLoader | None = None,  # for testing
) -> ComponentsInfo:
    config = model.config
    dataset_size = len(dataloader.dataset) if dataloader else 16

    attention_head_randoms = torch.rand(dataset_size, config.num_hidden_layers, config.num_attention_heads).to(model.device, non_blocking=True)
    attention_layer_randoms = torch.rand(dataset_size, config.num_hidden_layers).to(model.device, non_blocking=True)
    ffn_neuron_randoms = torch.rand(dataset_size, config.num_hidden_layers, config.intermediate_size).to(model.device, non_blocking=True)
    ffn_layer_randoms = torch.rand(dataset_size, config.num_hidden_layers).to(model.device, non_blocking=True)
    hidden_state_randoms = torch.rand(dataset_size, config.hidden_size).to(model.device, non_blocking=True)
    meta_randoms = torch.rand(dataset_size, 5).to(model.device, non_blocking=True)

    return ComponentsInfo(
        attention_head_randoms,
        attention_layer_randoms,
        ffn_neuron_randoms,
        ffn_layer_randoms,
        hidden_state_randoms,
        meta_randoms,
    )


def info_to_mean(
        components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[
        i.abs().mean(dim=0)
        for i in components_info
    ])


def info_to_max(
        components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[
        i.abs().max(dim=0).values
        for i in components_info
    ])


def info_to_fisher(
        components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[
        i.pow(2).sum(dim=0)
        for i in components_info
    ])


def info_to_entropy(
        components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[
        -(i * torch.log(i)).sum(dim=0)
        for i in components_info
    ])


def info_to_minus_entropy(
        components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[
        (i * torch.log(i)).sum(dim=0)
        for i in components_info
    ])
