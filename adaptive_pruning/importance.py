from __future__ import annotations

from typing import NamedTuple

import torch
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm
from transformers import PreTrainedModel

from adaptive_pruning.injections import (
    inject_attention_head_mask,
    inject_attention_layer_mask,
    inject_ffn_layer_mask,
    inject_ffn_neuron_mask,
    inject_hidden_state_mask,
)


class ComponentsInfo(NamedTuple):
    attention_heads_info: torch.Tensor  # [num_samples, num_layers, num_heads]
    attention_layers_info: torch.Tensor  # [num_samples, num_layers]
    ffn_neurons_info: torch.Tensor  # [num_samples, num_layers, inner_size]
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


def collect_mask_gradients(
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
    attention_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).to(
        device=model.device, dtype=model.dtype, non_blocking=True
    )
    ffn_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).to(
        device=model.device, dtype=model.dtype, non_blocking=True
    )
    ffn_layer_mask = torch.ones(config.num_hidden_layers).to(device=model.device, dtype=model.dtype, non_blocking=True)
    attention_layer_mask = torch.ones(config.num_hidden_layers).to(
        device=model.device, dtype=model.dtype, non_blocking=True
    )
    hidden_state_mask = torch.ones(config.hidden_size).to(device=model.device, dtype=model.dtype, non_blocking=True)

    # Requires grad to save it
    attention_head_mask.requires_grad_(True)
    ffn_neuron_mask.requires_grad_(True)
    ffn_layer_mask.requires_grad_(True)
    attention_layer_mask.requires_grad_(True)
    hidden_state_mask.requires_grad_(True)

    # Make meta-mask over all masks - 1 for each other mask
    meta_mask = torch.ones(5).to(model.device, dtype=model.dtype, non_blocking=True)
    meta_mask.requires_grad_(True)

    # apply masks to model
    handles: list[RemovableHandle] = [
        *inject_attention_head_mask(model, attention_head_mask, meta_mask[0]),
        *inject_attention_layer_mask(model, attention_layer_mask, meta_mask[1]),
        *inject_ffn_neuron_mask(model, ffn_neuron_mask, meta_mask[2]),
        *inject_ffn_layer_mask(model, ffn_layer_mask, meta_mask[3]),
        *inject_hidden_state_mask(model, hidden_state_mask, meta_mask[4]),
    ]

    attention_head_grads = torch.empty((0, config.num_hidden_layers, config.num_attention_heads)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    attention_layer_grads = torch.empty((0, config.num_hidden_layers)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    ffn_neuron_grads = torch.empty((0, config.num_hidden_layers, config.intermediate_size)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    ffn_layer_grads = torch.empty((0, config.num_hidden_layers)).to(model.device, dtype=model.dtype, non_blocking=True)
    hidden_state_grads = torch.empty((0, config.hidden_size)).to(model.device, dtype=model.dtype, non_blocking=True)
    meta_grads = torch.empty((0, 5)).to(model.device, dtype=model.dtype, non_blocking=True)

    for batch in batch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward(retain_graph=True)  # TODO: fix error in inject_attention_head_mask

        attention_head_grads = torch.cat([attention_head_grads, attention_head_mask.grad.detach().unsqueeze(0)], dim=0)
        attention_head_mask.grad = None

        attention_layer_grads = torch.cat(
            [attention_layer_grads, attention_layer_mask.grad.detach().unsqueeze(0)], dim=0
        )
        attention_layer_mask.grad = None

        ffn_neuron_grads = torch.cat([ffn_neuron_grads, ffn_neuron_mask.grad.detach().unsqueeze(0)], dim=0)
        ffn_neuron_mask.grad = None

        ffn_layer_grads = torch.cat([ffn_layer_grads, ffn_layer_mask.grad.detach().unsqueeze(0)], dim=0)
        ffn_layer_mask.grad = None

        hidden_state_grads = torch.cat([hidden_state_grads, hidden_state_mask.grad.detach().unsqueeze(0)], dim=0)
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
        attention_head_grads,
        attention_layer_grads,
        ffn_neuron_grads,
        ffn_layer_grads,
        hidden_state_grads,
        meta_grads,
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
    attention_head_activations = torch.empty((0, config.num_hidden_layers, config.num_attention_heads)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    attention_layer_activations = torch.empty((0, config.num_hidden_layers)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    ffn_neuron_activations = torch.empty((0, config.num_hidden_layers, config.intermediate_size)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    ffn_layer_activations = torch.empty((0, config.num_hidden_layers)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    hidden_state_activations = torch.empty((0, config.hidden_size)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    meta_activations = torch.empty((0, 5)).to(model.device, dtype=model.dtype, non_blocking=True)

    current_attention_head_activations = torch.empty((0, 0, config.num_attention_heads)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    current_attention_layer_activations = torch.empty((0, 0)).to(model.device, dtype=model.dtype, non_blocking=True)
    current_ffn_neuron_activations = torch.empty((0, 0, config.intermediate_size)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    current_ffn_layer_activations = torch.empty((0, 0)).to(model.device, dtype=model.dtype, non_blocking=True)
    current_hidden_state_activations = torch.empty((0, 0, config.hidden_size)).to(
        model.device, dtype=model.dtype, non_blocking=True
    )
    current_meta_activations = torch.empty((0, 5)).to(model.device, dtype=model.dtype, non_blocking=True)

    # Add hooks to collect activations
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):

        def hook_fn_attention(
            module: torch.nn.Module, input: torch.Tensor | tuple[torch.Tensor], output: torch.Tensor | None = None
        ):
            # output: tuple with 0: [batch_size, seq_len, hidden_size]
            # average over sequence length: [batch_size, hidden_size]
            # than split by heads: [batch_size, num_heads, head_size]
            # average over head_size: [batch_size, num_heads]
            nonlocal current_attention_head_activations, current_attention_layer_activations
            average_over_sequence = input[0].mean(dim=1)
            average_over_heads = average_over_sequence.view(
                -1, config.num_attention_heads, config.hidden_size // config.num_attention_heads
            ).mean(dim=2)
            # append to current_activations on the layers dimension (1)
            current_attention_head_activations = torch.cat(
                [current_attention_head_activations, average_over_heads.unsqueeze(1)], dim=1
            )
            current_attention_layer_activations = torch.cat(
                [current_attention_layer_activations, average_over_heads.mean(dim=1).unsqueeze(1)], dim=1
            )

        def hook_fn_ffn(
            module: torch.nn.Module, input: torch.Tensor | tuple[torch.Tensor], output: torch.Tensor | None = None
        ):
            # output: [batch_size, seq_len, intermediate_size]
            # average over sequence length: [batch_size, intermediate_size]
            nonlocal current_ffn_neuron_activations, current_ffn_layer_activations
            average_over_sequence = input[0].mean(dim=1)
            current_ffn_neuron_activations = torch.cat(
                [current_ffn_neuron_activations, average_over_sequence.unsqueeze(1)], dim=1
            )
            current_ffn_layer_activations = torch.cat(
                [current_ffn_layer_activations, average_over_sequence.mean(dim=1).unsqueeze(1)], dim=1
            )

        def hook_fn_hidden_states(
            module: torch.nn.Module, input: torch.Tensor | tuple[torch.Tensor], output: torch.Tensor | None = None
        ):
            # output: [batch_size, seq_len, hidden_size]
            # average over sequence length: [batch_size, hidden_size]
            nonlocal current_hidden_state_activations
            average_over_sequence = output[0].mean(dim=1)
            current_hidden_state_activations = torch.cat(
                [current_hidden_state_activations, average_over_sequence.unsqueeze(1)], dim=1
            )

        handles.append(layer.self_attn.o_proj.register_forward_pre_hook(hook_fn_attention))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook_fn_ffn))
        handles.append(layer.register_forward_hook(hook_fn_hidden_states))

    for batch in batch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        # update current_activations with 0 values to fill it
        batch_size = batch["input_ids"].shape[0]
        current_attention_head_activations = torch.zeros((batch_size, 0, config.num_attention_heads)).to(
            model.device, dtype=model.dtype, non_blocking=True
        )
        current_attention_layer_activations = torch.zeros((batch_size, 0)).to(
            model.device, dtype=model.dtype, non_blocking=True
        )
        current_ffn_neuron_activations = torch.zeros((batch_size, 0, config.intermediate_size)).to(
            model.device, dtype=model.dtype, non_blocking=True
        )
        current_ffn_layer_activations = torch.zeros((batch_size, 0)).to(
            model.device, dtype=model.dtype, non_blocking=True
        )
        current_hidden_state_activations = torch.zeros((batch_size, 0, config.hidden_size)).to(
            model.device, dtype=model.dtype, non_blocking=True
        )

        # forward pass
        _ = model(**batch)

        # cat current_activations to activations
        attention_head_activations = torch.cat(
            [attention_head_activations, current_attention_head_activations.detach()], dim=0
        )
        attention_layer_activations = torch.cat(
            [attention_layer_activations, current_attention_layer_activations.detach()], dim=0
        )
        ffn_neuron_activations = torch.cat([ffn_neuron_activations, current_ffn_neuron_activations.detach()], dim=0)
        ffn_layer_activations = torch.cat([ffn_layer_activations, current_ffn_layer_activations.detach()], dim=0)
        hidden_state_activations = torch.cat(
            [hidden_state_activations, current_hidden_state_activations.detach().mean(dim=1)], dim=0
        )

    meta_activations = torch.stack(
        [
            attention_head_activations.mean(dim=(1, 2)),
            attention_layer_activations.mean(dim=1),
            ffn_neuron_activations.mean(dim=(1, 2)),
            ffn_layer_activations.mean(dim=1),
            hidden_state_activations.mean(dim=1),
        ],
        dim=1,
    ).to(model.device, dtype=model.dtype, non_blocking=True)

    # remove hooks
    for handle in handles:
        handle.remove()

    return ComponentsInfo(
        attention_head_activations,
        attention_layer_activations,
        ffn_neuron_activations,
        ffn_layer_activations,
        hidden_state_activations,
        meta_activations,
    )


def collect_weight_magnitudes(
    model: PreTrainedModel,
    dataloader: DataLoader | None = None,  # for testing
) -> ComponentsInfo:
    config = model.config

    # Note: in case of weights we will have 1 sample (dim=0)
    attention_head_magnitudes = torch.empty((1, config.num_hidden_layers, config.num_attention_heads)).to(
        model.device, non_blocking=True
    )
    attention_layer_magnitudes = torch.empty((1, config.num_hidden_layers)).to(model.device, non_blocking=True)
    ffn_neuron_magnitudes = torch.empty((1, config.num_hidden_layers, config.intermediate_size)).to(
        model.device, non_blocking=True
    )
    ffn_layer_magnitudes = torch.empty((1, config.num_hidden_layers)).to(model.device, non_blocking=True)
    hidden_state_magnitudes = torch.empty((1, config.hidden_size)).to(model.device, non_blocking=True)
    meta_magnitudes = torch.empty((1, 5)).to(model.device, non_blocking=True)

    # Iterate through each layer of the model
    for layer_idx, layer in enumerate(model.model.layers):
        # Get attention heads magnitudes
        num_heads_per_key_value_group = config.num_attention_heads // config.num_key_value_heads
        mean_query_weight = layer.self_attn.q_proj.weight.mean(dim=1).view(config.num_attention_heads, -1).mean(dim=1)
        mean_key_weight = (
            layer.self_attn.k_proj.weight.mean(dim=1)
            .view(config.num_key_value_heads, -1)
            .mean(dim=1)
            .repeat_interleave(num_heads_per_key_value_group)
        )
        mean_value_weight = (
            layer.self_attn.v_proj.weight.mean(dim=1)
            .view(config.num_key_value_heads, -1)
            .mean(dim=1)
            .repeat_interleave(num_heads_per_key_value_group)
        )
        mean_output_weight = layer.self_attn.o_proj.weight.mean(dim=0).view(config.num_attention_heads, -1).mean(dim=1)
        mean_attention_weights = (mean_query_weight + mean_key_weight + mean_value_weight + mean_output_weight) / 4
        attention_head_magnitudes[0, layer_idx, :] = mean_attention_weights

        # Get attention layer magnitude
        attention_layer_magnitudes[0, layer_idx] = mean_attention_weights.mean().item()

        # Get FFN neuron magnitudes
        mean_ffn_gate_weights = layer.mlp.gate_proj.weight.mean(dim=1)
        mean_ffn_up_weights = layer.mlp.up_proj.weight.mean(dim=1)
        mean_ffn_down_weights = layer.mlp.down_proj.weight.mean(dim=0)
        mean_ffn_weights = (mean_ffn_gate_weights + mean_ffn_up_weights + mean_ffn_down_weights) / 3
        ffn_neuron_magnitudes[0, layer_idx, :] = mean_ffn_weights

        # Get FFN layer magnitude
        ffn_layer_magnitudes[0, layer_idx] = mean_ffn_weights.norm().item()

    # Get hidden state magnitudes
    # hidden_state_magnitudes[0, :] = torch.norm(model.model.embed_tokens.weight, dim=1)
    # TODO: weights hidden states

    # Compute meta magnitudes
    # meta_magnitudes  # [num_samples, 5] - attn_heads, attn_layers, ffn_neurons, ffn_layers, hidden_state

    meta_magnitudes[0, 0] = attention_head_magnitudes.mean().item()
    meta_magnitudes[0, 1] = attention_layer_magnitudes.mean().item()
    meta_magnitudes[0, 2] = ffn_neuron_magnitudes.mean().item()
    meta_magnitudes[0, 3] = ffn_layer_magnitudes.mean().item()
    meta_magnitudes[0, 4] = hidden_state_magnitudes.mean().item()

    return ComponentsInfo(
        attention_head_magnitudes,
        attention_layer_magnitudes,
        ffn_neuron_magnitudes,
        ffn_layer_magnitudes,
        hidden_state_magnitudes,
        meta_magnitudes,
    )


def collect_random_numbers(
    model: PreTrainedModel,
    dataloader: DataLoader | None = None,  # for testing
) -> ComponentsInfo:
    config = model.config
    dataset_size = len(dataloader.dataset) if dataloader else 16

    attention_head_randoms = torch.rand(dataset_size, config.num_hidden_layers, config.num_attention_heads).to(
        model.device, non_blocking=True
    )
    attention_layer_randoms = torch.rand(dataset_size, config.num_hidden_layers).to(model.device, non_blocking=True)
    ffn_neuron_randoms = torch.rand(dataset_size, config.num_hidden_layers, config.intermediate_size).to(
        model.device, non_blocking=True
    )
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
    return ComponentsImportance(*[i.abs().mean(dim=0) for i in components_info])


def info_to_max(
    components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[i.abs().max(dim=0).values for i in components_info])


def info_to_fisher(
    components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[i.pow(2).sum(dim=0) for i in components_info])


def info_to_entropy(
    components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[-(i * torch.log(i)).sum(dim=0) for i in components_info])


def info_to_minus_entropy(
    components_info: ComponentsInfo,
) -> ComponentsImportance:
    return ComponentsImportance(*[(i * torch.log(i)).sum(dim=0) for i in components_info])
