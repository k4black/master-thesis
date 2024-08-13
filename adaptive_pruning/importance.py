from __future__ import annotations

import copy
import math
from typing import Literal, NamedTuple

import torch
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedModel

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

    @classmethod
    def from_info(
        cls,
        components_info: ComponentsInfo,
        how_to_average: Literal["fisher_info", "mean", "max", "entropy", "minus_entropy"],
    ) -> ComponentsImportance:
        if how_to_average == "fisher_info":
            components_importance = info_to_fisher(components_info)
        elif how_to_average == "mean":
            components_importance = info_to_mean(components_info)
        elif how_to_average == "max":
            components_importance = info_to_max(components_info)
        elif how_to_average == "entropy":
            components_importance = info_to_entropy(components_info)
        elif how_to_average == "minus_entropy":
            components_importance = info_to_minus_entropy(components_info)
        else:
            assert False, f"Unknown how_to_average: {how_to_average}"
        return components_importance


class ComponentsToPrune(NamedTuple):
    attention_heads_to_prune: dict[int, list[int]]  # {layer: [heads]}
    attention_layers_to_prune: list[int]  # [layers]
    ffn_neurons_to_prune: dict[int, list[int]]  # {layer: [neurons]}
    ffn_layers_to_prune: list[int]  # [layers]
    hidden_states_to_prune: list[int]  # [hidden_states]

    @classmethod
    def from_importance(
        cls,
        components_importance: ComponentsImportance,
        pruning_ratio_target: float,
        pruning_components: list[str],  # ["attn_heads", "attn_layers", "ffn_layers", "ffn_neurons", "hidden_states"]
        round_to: int,
        is_uniform: bool,
        how_to_overlap: str,  # ["fixed", "relative", "meta"]
        config: PretrainedConfig,
        already_pruned_components: ComponentsToPrune | None = None,
    ) -> ComponentsToPrune:
        # get target ratios to prune
        attn_heads_ratio, attn_layers_ratio, ffn_neurons_ratio, ffn_layers_ratio, hidden_states_ratio = (
            get_components_ratios(pruning_ratio_target, pruning_components, how_to_overlap, components_importance)
        )

        # adjust rations to round_to and already pruned components
        total_attn_heads = config.num_hidden_layers * config.num_attention_heads
        total_attn_layers = config.num_hidden_layers
        total_ffn_neurons = config.num_hidden_layers * config.intermediate_size
        total_ffn_layers = config.num_hidden_layers
        total_hidden_states = config.hidden_size
        if already_pruned_components:
            current_attn_heads_ratio = (
                sum(len(v) for v in already_pruned_components.attention_heads_to_prune.values()) / total_attn_heads
            )
            current_attn_layers_ratio = len(already_pruned_components.attention_layers_to_prune) / total_attn_layers
            current_ffn_neurons_ratio = (
                sum(len(v) for v in already_pruned_components.ffn_neurons_to_prune.values()) / total_ffn_neurons
            )
            current_ffn_layers_ratio = len(already_pruned_components.ffn_layers_to_prune) / total_ffn_layers
            current_hidden_states_ratio = len(already_pruned_components.hidden_states_to_prune) / total_hidden_states

            attn_heads_ratio = max(attn_heads_ratio - current_attn_heads_ratio, 0)
            attn_layers_ratio = max(attn_layers_ratio - current_attn_layers_ratio, 0)
            ffn_neurons_ratio = max(ffn_neurons_ratio - current_ffn_neurons_ratio, 0)
            ffn_layers_ratio = max(ffn_layers_ratio - current_ffn_layers_ratio, 0)
            hidden_states_ratio = max(hidden_states_ratio - current_hidden_states_ratio, 0)

        # Skip pruned components
        if already_pruned_components:
            # Temporary put INF values to skip pruned components
            inf = float("inf")
            components_importance = copy.deepcopy(components_importance)
            for layer, heads in already_pruned_components.attention_heads_to_prune.items():
                components_importance.attention_heads_importance[layer, heads] = inf
            components_importance.attention_layers_importance[already_pruned_components.attention_layers_to_prune] = inf
            for layer, neurons in already_pruned_components.ffn_neurons_to_prune.items():
                components_importance.ffn_neurons_importance[layer, neurons] = inf
            components_importance.ffn_layers_importance[already_pruned_components.ffn_layers_to_prune] = inf
            components_importance.hidden_states_importance[already_pruned_components.hidden_states_to_prune] = inf

        # Select components to prune
        attention_layers_to_prune = select_to_prune_attention_layers(
            components_importance.attention_layers_importance,
            attn_layers_ratio,
        )
        head_size = config.hidden_size // config.num_attention_heads
        attention_heads_to_prune = select_to_prune_attention_heads(
            components_importance.attention_heads_importance,
            attn_heads_ratio,
            uniform_among_layers=is_uniform,
            key_value_group_size=config.num_attention_heads // config.num_key_value_heads,
            # round_to_heads=(round_to // head_size) or 1,
            round_to_heads=1,
        )
        attention_heads_to_prune = {
            layer: heads for layer, heads in attention_heads_to_prune.items() if layer not in attention_layers_to_prune
        }

        ffn_layers_to_prune = select_to_prune_ffn_layers(
            components_importance.ffn_layers_importance,
            ffn_layers_ratio,
        )
        neurons_to_prune = select_to_prune_ffn_neurons(
            components_importance.ffn_neurons_importance,
            ffn_neurons_ratio,
            uniform_among_layers=is_uniform,
            round_to=round_to,
            round_to_lower=True,
        )
        neurons_to_prune = {
            layer: neurons for layer, neurons in neurons_to_prune.items() if layer not in attention_layers_to_prune
        }
        hidden_states_to_prune = select_to_prune_hidden_states(
            components_importance.hidden_states_importance,
            hidden_states_ratio,
            round_to=round_to,
        )

        # Do not include empty dicts
        if all(len(v) == 0 for v in attention_heads_to_prune.values()):
            attention_heads_to_prune = {}
        if all(len(v) == 0 for v in neurons_to_prune.values()):
            neurons_to_prune = {}

        return ComponentsToPrune(
            attention_heads_to_prune,
            attention_layers_to_prune,
            neurons_to_prune,
            ffn_layers_to_prune,
            hidden_states_to_prune,
        )


def get_insert_pruning_masks(
    model: PreTrainedModel,
    require_grads: bool = True,
    dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], list[RemovableHandle]]:
    dtype = dtype or model.dtype

    # create masks for the model
    pruning_masks = {
        "attn_heads": torch.ones(
            model.config.num_hidden_layers,
            model.config.num_attention_heads,
            device=model.device,
            dtype=dtype,
            requires_grad=require_grads,
        ),
        "ffn_neurons": torch.ones(
            model.config.num_hidden_layers,
            model.config.intermediate_size,
            device=model.device,
            dtype=dtype,
            requires_grad=require_grads,
        ),
        "ffn_layers": torch.ones(
            model.config.num_hidden_layers, device=model.device, dtype=dtype, requires_grad=require_grads
        ),
        "attn_layers": torch.ones(
            model.config.num_hidden_layers, device=model.device, dtype=dtype, requires_grad=require_grads
        ),
        "hidden_states": torch.ones(
            model.config.hidden_size, device=model.device, dtype=dtype, requires_grad=require_grads
        ),
        "meta": torch.ones(5, device=model.device, dtype=dtype, requires_grad=require_grads),
    }

    for name, mask in pruning_masks.items():
        mask.requires_grad_(require_grads)

    # insert the hooks
    pruning_masks_hooks = [
        *inject_attention_head_mask(model, pruning_masks["attn_heads"], pruning_masks["meta"][0]),
        *inject_attention_layer_mask(model, pruning_masks["attn_layers"], pruning_masks["meta"][1]),
        *inject_ffn_neuron_mask(model, pruning_masks["ffn_neurons"], pruning_masks["meta"][2]),
        *inject_ffn_layer_mask(model, pruning_masks["ffn_layers"], pruning_masks["meta"][3]),
        *inject_hidden_state_mask(model, pruning_masks["hidden_states"], pruning_masks["meta"][4]),
    ]

    return pruning_masks, pruning_masks_hooks


def collect_mask_gradients(
    model: PreTrainedModel,
    dataloader: DataLoader,
    tuple_pruning_masks_hooks: tuple[dict[str, torch.Tensor], list[RemovableHandle]] | None = None,
    *,
    verbose: bool = True,
) -> ComponentsInfo:
    batch_iterator = tqdm(dataloader, total=len(dataloader), desc="Collecting grads") if verbose else dataloader
    config = model.config

    # Insert masks if not provided
    if tuple_pruning_masks_hooks is None:
        pruning_masks, pruning_masks_hooks = get_insert_pruning_masks(model)
    else:
        pruning_masks, pruning_masks_hooks = tuple_pruning_masks_hooks

    # Create tensors to store the gradients
    gradient_collectors = {
        name: torch.empty((0, *mask.shape), device=model.device, dtype=model.dtype, requires_grad=False)
        for name, mask in pruning_masks.items()
    }

    # Disable grads for other model
    model.eval()
    # enable grad for masks
    for name in pruning_masks.keys():
        pruning_masks[name].requires_grad_(True)

    for batch in batch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(model.device, non_blocking=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward(retain_graph=True)  # TODO: fix error in inject_attention_head_mask

        for name in gradient_collectors:
            gradient_collectors[name] = torch.cat(
                [gradient_collectors[name], pruning_masks[name].grad.detach().unsqueeze(0)], dim=0
            )
            pruning_masks[name].detach_()

    # clear graph, clear grads
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = None
    for mask in pruning_masks.values():
        mask.requires_grad_(False)
        mask.detach_()
        mask.grad = None
    # remove masks from the model
    if tuple_pruning_masks_hooks is None:
        for handle in pruning_masks_hooks:
            handle.remove()
        if not tuple_pruning_masks_hooks:
            for name in pruning_masks.keys():
                pruning_masks[name].requires_grad_(False)
                pruning_masks[name].detach_()
                gradient_collectors[name].detach_()
            del pruning_masks

    # clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # disable grad
    model.train()

    return ComponentsInfo(
        gradient_collectors["attn_heads"].detach(),
        gradient_collectors["attn_layers"].detach(),
        gradient_collectors["ffn_neurons"].detach(),
        gradient_collectors["ffn_layers"].detach(),
        gradient_collectors["hidden_states"].detach(),
        gradient_collectors["meta"].detach(),
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


def select_to_prune_attention_heads(
    importance_scores: torch.Tensor,
    percent_heads_to_prune: float,
    uniform_among_layers: bool = False,
    key_value_group_size: int = 1,
    round_to_heads: int = 1,
    keep_at_least_one_head: bool = False,
) -> dict[int, list[int]]:
    """
    Select least-k attention heads based on the importance scores.
    Keep at least 1 attention head per layer

    Note: prune heads regardless of the layer, so remove the least important head among the whole model
    i.e. can prune all heads for layer 1 and none in layer 2

    :param importance_scores: The importance scores of the attention heads [num_hidden_layers, num_attention_heads]
    :param percent_heads_to_prune: The percentage of attention heads to keep
    :param uniform_among_layers: If True, prune the same number of heads from each layer
    :param key_value_group_size: The number of attention heads to group together for key and value projections; 1 means no grouping
    :param round_to_heads: The number of heads to group together for pruning
        TODO: Think to group K or QV heads. For now, round grouped heads number
    :param keep_at_least_one_head: If True, keep at least 1 head per layer
    :return: A dictionary with the layer indices as keys and a list of head indices to prune as values
    """
    assert 0 <= percent_heads_to_prune <= 1, "percent_heads_to_prune should be in [0, 1]"
    assert (
        1 <= key_value_group_size <= importance_scores.size(1)
    ), "key_value_group_size should be in [1, num_attention_heads]"

    # shrink the importance scores to the grouped size by averaging
    if key_value_group_size != 1:
        importance_scores = importance_scores.view(
            importance_scores.size(0), importance_scores.size(1) // key_value_group_size, key_value_group_size
        ).mean(dim=-1)

    heads_grouped_to_prune = {}
    num_layers, num_heads_grouped = importance_scores.size()

    if uniform_among_layers:
        num_heads_grouped_to_prune = int(num_heads_grouped * percent_heads_to_prune)
        num_heads_grouped_to_prune = round(num_heads_grouped_to_prune / round_to_heads) * round_to_heads
        num_heads_grouped_to_prune = min(num_heads_grouped_to_prune, num_heads_grouped)

        for layer_index in range(num_layers):
            # sort heads by importance
            importance_scores_layer = importance_scores[layer_index]
            _, sorted_indices = importance_scores_layer.sort()
            heads_to_prune_layer = sorted_indices[:num_heads_grouped_to_prune].tolist()
            heads_grouped_to_prune[layer_index] = heads_to_prune_layer

    else:
        num_heads_grouped_to_prune_total = int(num_layers * num_heads_grouped * percent_heads_to_prune)

        # sort heads by importance
        importance_scores_flatten = importance_scores.view(-1)
        _, sorted_indices = importance_scores_flatten.sort()
        heads_to_prune_flatten = sorted_indices[:num_heads_grouped_to_prune_total]

        # convert to layer-head indices
        for head_index in heads_to_prune_flatten:
            head_index = int(head_index.item())
            layer_index = head_index // num_heads_grouped
            head_index = head_index % num_heads_grouped
            heads_grouped_to_prune.setdefault(layer_index, []).append(head_index)

        # round to the nearest round_to_heads for each layer (drop excess heads)
        for layer, heads in heads_grouped_to_prune.items():
            num_heads_to_prune_layer = len(heads)
            num_heads_to_prune_layer = math.floor(num_heads_to_prune_layer / round_to_heads) * round_to_heads
            heads_grouped_to_prune[layer] = heads[:num_heads_to_prune_layer]

    # keep at least 1 head per layer (or round_to_heads), remove unused heads_to_prune lists
    if keep_at_least_one_head:
        for layer in list(heads_grouped_to_prune.keys()):
            if len(heads_grouped_to_prune[layer]) == num_heads_grouped:
                heads_grouped_to_prune[layer] = heads_grouped_to_prune[layer][: num_heads_grouped - round_to_heads]
            if len(heads_grouped_to_prune[layer]) == 0:
                del heads_grouped_to_prune[layer]

    # expand the grouped heads to the original size
    if key_value_group_size == 1:
        heads_to_prune = heads_grouped_to_prune
    else:
        heads_to_prune_expanded = {}
        for layer, heads in heads_grouped_to_prune.items():
            heads_expanded = [i * key_value_group_size + j for i in heads for j in range(key_value_group_size)]
            heads_to_prune_expanded[layer] = heads_expanded
        heads_to_prune = heads_to_prune_expanded

    return heads_to_prune


def select_to_prune_attention_layers(
    importance_scores: torch.Tensor,
    percent_layers_to_prune: float,
    # skip_layers: list[int] | None = None,
) -> list[int]:
    """
    Select least-k attention layers based on the importance scores.

    :param importance_scores: The importance scores of the attention layers [num_hidden_layers]
    :param percent_layers_to_prune: The percentage of attention layers to keep
    :return: A list of layer indices to prune
    """
    assert 0 <= percent_layers_to_prune <= 1, "percent_layers_to_prune should be in [0, 1]"

    num_layers = importance_scores.size(0)
    num_layers_to_prune = int(num_layers * percent_layers_to_prune)

    _, sorted_indices = importance_scores.sort()
    # if skip_layers:
    #     sorted_indices = torch.tensor([i for i in sorted_indices if i not in skip_layers])
    layers_to_prune = sorted_indices[:num_layers_to_prune].tolist()

    return layers_to_prune


def select_to_prune_ffn_neurons(
    importance_scores: torch.Tensor,
    percent_neurons_to_prune: float,
    uniform_among_layers: bool = False,
    round_to: int = 1,
    round_to_lower: bool = False,
) -> dict[int, list[int]]:
    """
    Select least-k feed forward neurons based on the importance scores.

    :param importance_scores: The importance scores of the feed forward neurons [num_hidden_layers, intermediate_size]
    :param percent_neurons_to_prune: The percentage of feed forward neurons to keep
    :param uniform_among_layers: If True, prune the same number of neurons from each layer
    :param round_to: The number of neurons to group together for pruning (round to the nearest round_to) for gpu opts
    :param round_to_lower: If True, round to the lower number of neurons
    :return: A dictionary with the layer indices as keys and a list of neuron indices to prune as values
    """
    assert 0 <= percent_neurons_to_prune <= 1, "percent_neurons_to_prune should be in [0, 1]"

    neurons_to_prune = {}
    num_layers, num_neurons = importance_scores.size()

    if uniform_among_layers:
        num_neurons_to_prune = int(num_neurons * percent_neurons_to_prune)
        if round_to_lower:
            num_neurons_to_prune = math.floor(num_neurons_to_prune / round_to) * round_to
        else:
            num_neurons_to_prune = round(num_neurons_to_prune / round_to) * round_to
        num_neurons_to_prune = min(num_neurons_to_prune, num_neurons)

        for layer_index in range(num_layers):
            # sort neurons by importance
            importance_scores_layer = importance_scores[layer_index]
            _, sorted_indices = importance_scores_layer.sort()
            neurons_to_prune_layer = sorted_indices[:num_neurons_to_prune].tolist()
            neurons_to_prune[layer_index] = neurons_to_prune_layer

    else:
        num_neurons_to_prune = int(num_layers * num_neurons * percent_neurons_to_prune)

        # sort neurons by importance
        importance_scores_flatten = importance_scores.view(-1)
        _, sorted_indices = importance_scores_flatten.sort()
        neurons_to_prune_flatten = sorted_indices[:num_neurons_to_prune]

        # convert to layer-neuron indices
        for neuron_index in neurons_to_prune_flatten:
            neuron_index = int(neuron_index.item())
            layer_index = neuron_index // num_neurons
            neuron_index = neuron_index % num_neurons
            neurons_to_prune.setdefault(layer_index, []).append(neuron_index)

        # round to the nearest round_to for each layer (drop excess neurons)
        for layer, neurons in neurons_to_prune.items():
            num_neurons_to_prune_layer = len(neurons)
            num_neurons_to_prune_layer = math.floor(num_neurons_to_prune_layer / round_to) * round_to
            neurons_to_prune[layer] = neurons[:num_neurons_to_prune_layer]

    return neurons_to_prune


def select_to_prune_ffn_layers(
    importance_scores: torch.Tensor,
    percent_layers_to_prune: float,
) -> list[int]:
    """
    Select least-k feed forward layers based on the importance scores.

    :param importance_scores: The importance scores of the feed forward layers [num_hidden_layers]
    :param percent_layers_to_prune: The percentage of feed forward layers to keep
    :return: A list of layer indices to prune
    """
    assert 0 <= percent_layers_to_prune <= 1, "percent_layers_to_prune should be in [0, 1]"

    num_layers = importance_scores.size(0)
    num_layers_to_prune = int(num_layers * percent_layers_to_prune)

    _, sorted_indices = importance_scores.sort()
    layers_to_prune = sorted_indices[:num_layers_to_prune].tolist()

    return layers_to_prune


def select_to_prune_hidden_states(
    importance_scores: torch.Tensor,
    percent_neurons_to_prune: float,
    round_to: int = 1,
) -> list[int]:
    """
    Select least-k neurons based on the importance scores.

    :param importance_scores: The importance scores of the hidden states [hidden_size]
    :param percent_neurons_to_prune: The percentage of hidden states to keep
    :return: A list of neuron indices to prune
    """
    assert 0 <= percent_neurons_to_prune <= 1, "percent_neurons_to_prune should be in [0, 1]"

    num_neurons = importance_scores.size(0)
    num_neurons_to_prune = int(num_neurons * percent_neurons_to_prune)
    num_neurons_to_prune = round(num_neurons_to_prune / round_to) * round_to
    num_neurons_to_prune = min(num_neurons_to_prune, num_neurons)

    _, sorted_indices = importance_scores.sort()
    neurons_to_prune = sorted_indices[:num_neurons_to_prune].tolist()

    return neurons_to_prune


def get_components_ratios(
    pruning_ratio: float,
    pruning_components: list[str],  # ["attn_heads", "attn_layers", "ffn_layers", "ffn_neurons", "hidden_states"]
    how_to_overlap: str,  # ["fixed", "relative", "meta"]
    components_importance: ComponentsImportance | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Calculate the ratios of pruning for each component based on the pruning ratio and components to prune.

    :param pruning_ratio: The overall pruning ratio
    :param pruning_components: The components to prune
    :param how_to_overlap: The strategy to overlap the pruning ratios
    :return: The ratios of pruning for each component
    """

    attention_heads_ratio = pruning_ratio if "attn_heads" in pruning_components else 0.0
    attention_layers_ratio = pruning_ratio if "attn_layers" in pruning_components else 0.0
    ffn_neurons_ratio = pruning_ratio if "ffn_neurons" in pruning_components else 0.0
    ffn_layers_ratio = pruning_ratio if "ffn_layers" in pruning_components else 0.0
    hidden_states_ratio = pruning_ratio if "hidden_states" in pruning_components else 0.0

    if how_to_overlap == "fixed":
        return attention_heads_ratio, attention_layers_ratio, ffn_neurons_ratio, ffn_layers_ratio, hidden_states_ratio

    elif how_to_overlap == "fixed_x1_x15_x05":
        # x1 attn heads, x1.5 neurons, x0.5 hidden states
        return attention_heads_ratio * 1, 0, ffn_neurons_ratio * 1.5, 0, hidden_states_ratio * 0.5

    elif how_to_overlap == "fixed_x1_x15_x01":
        return attention_heads_ratio * 1, 0, ffn_neurons_ratio * 1.5, 0, hidden_states_ratio * 0.1

    elif how_to_overlap == "relative":
        assert components_importance is not None, "components_importance should be provided for relative importance"
        relative_importance = torch.tensor(
            [
                components_importance.attention_heads_importance.cpu().mean().item(),
                # components_importance.attention_layers_importance.cpu().mean().item(),
                components_importance.ffn_neurons_importance.cpu().mean().item(),
                # components_importance.ffn_layers_importance.cpu().mean().item(),
                components_importance.hidden_states_importance.cpu().mean().item(),
            ]
        )
        relative_importance = 1 / relative_importance
        relative_importance /= relative_importance.sum()

        # normalize the ratio to have mean ratio as pruning_ratio
        pruning_ratios = relative_importance * pruning_ratio * 3
        pruning_ratios = torch.clip(pruning_ratios, 0, 1)

        print(f"relative_importance: {relative_importance}")
        print(f"pruning_ratio: {pruning_ratio}")
        print(f"pruning_ratios: {pruning_ratios}")
        return pruning_ratios[0].item(), 0.0, pruning_ratios[1].item(), 0.0, pruning_ratios[2].item()

    elif how_to_overlap == "relative_per_param":
        # TODO: Fix
        # Use actual relative importance to calculate pruning ratio
        # Use number of possible pruning parameters to calculate pruning ratio
        # adjust for number of params in the component
        _num_q_per_kv = config.num_attention_heads // config.num_key_value_heads
        _single_head_size = config.hidden_size // config.num_attention_heads
        # k + v of the same size and q and output of the same size
        num_parameters_attention_head_group = (
            2 * _single_head_size * config.hidden_size + 2 * _single_head_size * _num_q_per_kv * config.hidden_size
        )
        num_parameters_attention_layer = num_parameters_attention_head_group * config.num_key_value_heads  # + bias
        # gate, input, output
        num_parameters_ffn_neuron = 3 * config.hidden_size
        num_parameters_ffn_layer = num_parameters_ffn_neuron * config.intermediate_size  # + bias
        # hidden state = emb + 2 norms, attention, ffn
        num_parameters_hidden_state = config.vocab_size + config.num_hidden_layers * (
            2
            + 2 * _single_head_size * config.num_key_value_heads
            + 2 * _single_head_size * config.num_attention_heads
            + 3 * config.intermediate_size
        )
        total_parameters = num_parameters_hidden_state * config.hidden_size + config.num_hidden_layers * (  # noqa: F841
            num_parameters_attention_layer + num_parameters_ffn_layer
        )

        relative_importance = torch.tensor(
            [
                components_importance.attention_heads_importance.cpu().mean().item()
                / num_parameters_attention_head_group,
                components_importance.attention_layers_importance.cpu().mean().item() / num_parameters_attention_layer,
                components_importance.ffn_neurons_importance.cpu().mean().item() / num_parameters_ffn_neuron,
                components_importance.ffn_layers_importance.cpu().mean().item() / num_parameters_ffn_layer,
                components_importance.hidden_states_importance.cpu().mean().item() / num_parameters_hidden_state,
            ]
        )
        relative_pruning_coefficient = torch.softmax(-1 * relative_importance, dim=0) * relative_importance.shape[0]
        print("Relative importance", relative_importance)
        print("Relative pruning coefficient", relative_pruning_coefficient)

        attention_heads_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[0].item()
            if do_prune_attention_heads or do_prune_attention_heads_uniform
            else 0
        )
        attention_layers_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[1].item() if do_prune_attention_layers else 0
        )
        ffn_neurons_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[2].item()
            if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform
            else 0
        )
        ffn_layers_pruning_ratio = pruning_ratio * relative_pruning_coefficient[3].item() if do_prune_ffn_layers else 0
        hidden_states_pruning_ratio = (
            pruning_ratio * relative_pruning_coefficient[4].item() if do_prune_hidden_states else 0
        )

    elif how_to_overlap == "meta":
        assert components_importance is not None, "components_importance should be provided for meta importance"
        meta_importance = components_importance.meta_importance.cpu()
        meta_importance = torch.tensor(
            [
                meta_importance[0].mean().item(),
                # meta_importance[1].mean().item(),
                meta_importance[2].mean().item(),
                # meta_importance[3].mean().item(),
                meta_importance[4].mean().item(),
            ]
        )
        meta_importance = 1 / meta_importance
        meta_importance /= meta_importance.sum()

        # normalize the ratio to have mean ratio as pruning_ratio
        pruning_ratios = meta_importance * pruning_ratio * 5
        pruning_ratios = torch.clip(pruning_ratios, 0, 1)

        print(f"meta_importance: {meta_importance}")
        print(f"pruning_ratio: {pruning_ratio}")
        print(f"pruning_ratios: {pruning_ratios}")
        return pruning_ratios[0].item(), 0.0, pruning_ratios[1].item(), 0.0, pruning_ratios[2].item()

    else:
        assert False, f"Unknown how_to_overlap: {how_to_overlap}"
