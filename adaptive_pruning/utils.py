from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import tabulate
import torch
from calflops import calculate_flops
from transformers import PreTrainedModel, PreTrainedTokenizer


if TYPE_CHECKING:
    from .importance import ComponentsImportance, ComponentsInfo


def count_flops_macs_params(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    max_seq_length: int = 128,
    *,
    print_results: bool = True,
) -> tuple[int, int, int]:
    max_seq_length = max_seq_length or tokenizer.model_max_length

    flops, macs, lib_params = calculate_flops(
        model=model,
        transformer_tokenizer=tokenizer,
        input_shape=(batch_size, max_seq_length),
        include_backPropagation=False,
        print_results=False,
        print_detailed=False,
        output_as_string=False,
        output_precision=4,
    )
    params = count_total_parameters(model, require_grad=None)
    assert lib_params == params, f"Library params: {lib_params} != Custom params: {params}"

    if print_results:
        print(
            f"FLOPs: {format_number(flops)}\tMACs: {format_number(macs)}\tParams: {format_number(params)} ({params}) \n"
        )

    return flops, macs, params


def count_zero_parameters(
    module: torch.nn.Module,
    require_grad: bool | None = None,
) -> int:
    if require_grad is None:
        return sum(torch.sum(param == 0).item() for param in module.parameters())
    else:
        return sum(torch.sum(param == 0).item() for param in module.parameters() if param.requires_grad == require_grad)


def count_nonzero_parameters(
    module: torch.nn.Module,
    require_grad: bool | None = None,
) -> int:
    if require_grad is None:
        return sum(torch.sum(param != 0).item() for param in module.parameters())
    else:
        return sum(torch.sum(param != 0).item() for param in module.parameters() if param.requires_grad == require_grad)


def count_total_parameters(
    module: torch.nn.Module,
    require_grad: bool | None = None,
) -> int:
    if require_grad is None:
        return sum(p.numel() for p in module.parameters())
    else:
        return sum(p.numel() for p in module.parameters() if p.requires_grad == require_grad)


def measure_original_model_stats(model: PreTrainedModel, print_results: bool = False) -> dict[str, Any]:
    """
    Measure the number of parameters in the model and its layers.
    Also includes the number of parameters in the attention heads and feedforward layers.

    :param model: model to measure, only llama models are supported
    :param print_results: whether to print the results
    :return: dictionary with the number of parameters in the model and its layers
    """
    base_model = model.base_model if hasattr(model, "base_model") else model
    assert (
        "llama" in base_model.config.model_type.lower()
    ), f"Only llama models are supported, got {base_model.config.model_type}"

    model_stats = {
        "total": count_total_parameters(model),
        "base": count_total_parameters(base_model),
    }
    for i, layer in enumerate(base_model.layers):
        model_stats[f"layer_{i}"] = count_total_parameters(layer)
        # attention heads
        model_stats[f"layer_{i}/attn"] = count_total_parameters(layer.self_attn)
        # feedforward
        model_stats[f"layer_{i}/ffn"] = count_total_parameters(layer.mlp)

    if print_results:
        headers = ["Layer", "#Params"]
        rows = [[key.replace("layer_", ""), format_number(value)] for key, value in model_stats.items()]
        print(tabulate.tabulate(rows, headers=headers, tablefmt="pretty", colalign=("left",)))
        sys.stdout.flush()

    return model_stats


def measure_pruned_model_stats(
    model: PreTrainedModel, original_model_stats: dict[str, Any] | None = None, print_results: bool = False
) -> dict[str, Any]:
    """
    Measure the number of parameters in the model and its layers.
    Also includes the number of parameters in the attention heads and feedforward layers.
    Calculates stats with respect to the original model stats if provided.

    :param model: model to measure, only llama models are supported
    :param original_model_stats: original model stats to compare with
    :param print_results: whether to print the results
    :return: dictionary with the number of parameters in the model and its layers
    """
    base_model = model.base_model if hasattr(model, "base_model") else model
    assert (
        "llama" in base_model.config.model_type.lower()
    ), f"Only llama models are supported, got {base_model.config.model_type}"

    # Check total sparsity
    sparsity_stats = {
        "total": {
            "num_original_parameters": original_model_stats["total"] if original_model_stats else None,
            "num_parameters": count_total_parameters(model),
            "num_zero_parameters": count_zero_parameters(model),
            "num_nonzero_parameters": count_nonzero_parameters(model),
        },
        "base": {
            "num_original_parameters": original_model_stats["base"] if original_model_stats else None,
            "num_parameters": count_total_parameters(base_model),
            "num_zero_parameters": count_zero_parameters(base_model),
            "num_nonzero_parameters": count_nonzero_parameters(base_model),
        },
    }
    # Check sparsity by layer (unstructured and structured)
    for i, layer in enumerate(base_model.layers):
        sparsity_stats[f"layer_{i}"] = {
            "num_original_parameters": original_model_stats[f"layer_{i}"] if original_model_stats else None,
            "num_parameters": count_total_parameters(layer),
            "num_zero_parameters": count_zero_parameters(layer),
            "num_nonzero_parameters": count_nonzero_parameters(layer),
        }
        # attention heads
        sparsity_stats[f"layer_{i}/attn"] = {
            "num_original_parameters": original_model_stats[f"layer_{i}/attn"] if original_model_stats else None,
            "num_parameters": count_total_parameters(layer.self_attn),
            "num_zero_parameters": count_zero_parameters(layer.self_attn),
            "num_nonzero_parameters": count_nonzero_parameters(layer.self_attn),
        }
        # feedforward
        sparsity_stats[f"layer_{i}/ffn"] = {
            "num_original_parameters": original_model_stats[f"layer_{i}/ffn"] if original_model_stats else None,
            "num_parameters": count_total_parameters(layer.mlp),
            "num_zero_parameters": count_zero_parameters(layer.mlp),
            "num_nonzero_parameters": count_nonzero_parameters(layer.mlp),
        }

    # Add percentage columns
    for key, stats in sparsity_stats.items():
        stats["percentage_original_pruned"] = (
            (stats["num_original_parameters"] - stats["num_parameters"]) / stats["num_original_parameters"] * 100
            if stats["num_original_parameters"]
            else None
        )
        stats["percentage_original_zero"] = (
            stats["num_zero_parameters"] / stats["num_original_parameters"] * 100
            if stats["num_original_parameters"]
            else None
        )
        stats["percentage_original_pruned_or_zero"] = (
            (stats["num_original_parameters"] - stats["num_parameters"] + stats["num_zero_parameters"])
            / stats["num_original_parameters"]
            * 100
            if stats["num_original_parameters"]
            else None
        )
        stats["percentage_zero"] = stats["num_zero_parameters"] / stats["num_parameters"] * 100

    if print_results:
        headers = [
            "Layer",
            "#Params\n(Original)",
            "#Params\n(Pruned)",
            "#Params\n(Zero)",
            "#Params\n(Nonzero)",
            "%Pruned",
            "%Zero",
            "%Pruned or Zero",
            "%Zero\n(Current)",
        ]
        table = [
            [
                "_" + fkey if "/" in (fkey := key.replace("layer_", "")) else fkey,
                (
                    format_number(values["num_original_parameters"])
                    if values["num_original_parameters"] is not None
                    else "-"
                ),
                format_number(values["num_parameters"]),
                format_number(values["num_zero_parameters"]),
                format_number(values["num_nonzero_parameters"]),
                (
                    f"{values['percentage_original_pruned']:.2f}%"
                    if values["percentage_original_pruned"] is not None
                    else "-"
                ),
                f"{values['percentage_original_zero']:.2f}%" if values["percentage_original_zero"] is not None else "-",
                (
                    f"{values['percentage_original_pruned_or_zero']:.2f}%"
                    if values["percentage_original_pruned_or_zero"] is not None
                    else "-"
                ),
                f"{values['percentage_zero']:.2f}%",
            ]
            for key, values in sparsity_stats.items()
        ]
        print(tabulate.tabulate(table, headers=headers, tablefmt="pretty", colalign=("left",)))
        sys.stdout.flush()

    return sparsity_stats


def print_components_info_importance(components_info_importance: ComponentsInfo | ComponentsImportance) -> None:
    def _print_value(value: torch.Tensor) -> str:
        return f"{value.mean().item():.4f} ± {value.std().item():.2f}"

    headers = ["Layer", "Attn heads", "Attn layers", "FFN neurons", "FFN layers", "Hidden states"]
    table = []
    try:
        table.append(
            [
                "total",
                _print_value(components_info_importance.attention_heads_info),
                _print_value(components_info_importance.attention_layers_info),
                _print_value(components_info_importance.ffn_neurons_info),
                _print_value(components_info_importance.ffn_layers_info),
                _print_value(components_info_importance.hidden_states_info),
            ]
        )
        for layer in range(components_info_importance.attention_heads_info.size(1)):
            table.append(
                [
                    layer,
                    _print_value(components_info_importance.attention_heads_info[:, layer, :]),
                    _print_value(components_info_importance.attention_layers_info[:, layer]),
                    _print_value(components_info_importance.ffn_neurons_info[:, layer, :]),
                    _print_value(components_info_importance.ffn_layers_info[:, layer]),
                    "-",
                ]
            )
        meta = components_info_importance.meta_info
    except AttributeError:
        table.append(
            [
                "total",
                _print_value(components_info_importance.attention_heads_importance),
                _print_value(components_info_importance.attention_layers_importance),
                _print_value(components_info_importance.ffn_neurons_importance),
                _print_value(components_info_importance.ffn_layers_importance),
                _print_value(components_info_importance.hidden_states_importance),
            ]
        )
        for layer in range(components_info_importance.attention_heads_importance.size(0)):
            table.append(
                [
                    layer,
                    _print_value(components_info_importance.attention_heads_importance[layer, :]),
                    _print_value(components_info_importance.attention_layers_importance[layer]),
                    _print_value(components_info_importance.ffn_neurons_importance[layer, :]),
                    _print_value(components_info_importance.ffn_layers_importance[layer]),
                    _print_value(components_info_importance.hidden_states_importance),
                ]
            )
        meta = components_info_importance.meta_importance[None, :]
    print(tabulate.tabulate(table, headers=headers, tablefmt="pretty"))
    headers = ["Attn heads", "Attn layers", "FFN neurons", "FFN layers", "Hidden states"]
    table = [[f"{value.mean().item():.4f}" for value in meta.T]]
    print("Meta:")
    print(tabulate.tabulate(table, headers=headers, tablefmt="pretty", colalign=("left",)))
    sys.stdout.flush()


def format_number(number: int) -> str:
    """Format a number with K/M/B suffixes.

    :param number: The number to format
    :return: The formatted number
    """
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    elif abs(number) >= 1_000_000:
        formatted_number = round(number / 1_000_000, 1)
        return f"{formatted_number:.1f}M" if formatted_number < 1000 else f"{formatted_number / 1000:.1f}B"
    elif abs(number) >= 1_000:
        formatted_number = round(number / 1_000, 1)
        return f"{formatted_number:.1f}K" if formatted_number < 1000 else f"{formatted_number / 1000:.1f}M"
    else:
        return str(number)


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.cpu().numpy().tolist()
