from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any

import tabulate
import torch
from calflops import calculate_flops
from peft import LoraModel, PeftModelForCausalLM
from transformers import LlamaForCausalLM, PreTrainedModel, PreTrainedTokenizer


if TYPE_CHECKING:
    from .importance import ComponentsImportance, ComponentsInfo


def get_base_model(model: PreTrainedModel) -> PreTrainedModel:
    if isinstance(model, PeftModelForCausalLM):
        return get_base_model(model.base_model)
    if isinstance(model, LoraModel):
        return get_base_model(model.model)
    if isinstance(model, LlamaForCausalLM):
        return model.base_model
    return model.base_model if hasattr(model, "base_model") else model


def count_flops_macs_params(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 1,
    max_seq_length: int = 128,
    *,
    print_results: bool = True,
) -> tuple[int, int, int, int]:
    max_seq_length = max_seq_length or tokenizer.model_max_length

    try:
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
    except ValueError as e:
        print("FLOPs calculation failed, using 0:", e)
        flops, macs, lib_params = 0, 0, 0
    params = count_total_parameters(model, require_grad=None)
    zero_params = count_zero_parameters(model, require_grad=None)
    if lib_params != params:
        print(f"WARNING: The library calculated {lib_params} parameters, but the model has {params} parameters.")
        print("total required_grad=None  params:", params)
        print("total required_grad=True  params:", count_total_parameters(model, require_grad=True))
        print("total required_grad=False params:", count_total_parameters(model, require_grad=False))

    if print_results:
        print(
            f"FLOPs: {format_number(flops)}\t"
            f"MACs: {format_number(macs)}\t"
            f"Params: {format_number(params)} ({params})\t"
            f"Zero params: {format_number(zero_params)} ({zero_params/params*100:.2f}%)"
        )

    return flops, macs, params, zero_params


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


def measure_model_stats(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    original_model_stats: dict[str, Any] | None = None,
    print_results: bool = False,
) -> tuple[dict[str | int, Any], dict[str, int]]:
    """
    Measure the number of parameters in the model and its layers.
    Collect pruned
    When original_model_stats are provided, calculate the difference in the number of parameters.
    This func can be use_d to measure both original model and then pruned model stats.
    :param model: LLAMA Model to measure
    :param original_model_stats: optional original model stats to compare with
    :param print_results: whether to print the results
    :return: dict of the keys of ['total', 0, 1, ..., num_layers]. Each key contains the dict for the layer stats:
        ['n_params', 'n_zero_params', 'attn_heads_n_params', 'attn_heads_n_zero_params', 'ffn_n_params', 'ffn_n_zero_params']
        if original_model_stats are provided, the dict will also contain the 'X_pruned_percentage' keys
        and second dict with total - flops, macs, params, zero_params
    """
    base_model = get_base_model(model)
    assert (
        "llama" in base_model.config.model_type.lower()
    ), f"Only llama models are supported, got {base_model.config.model_type}"

    # Check sparsity by layer (unstructured and structured)
    sparsity_stats = {}
    for i, layer in enumerate(base_model.layers):
        sparsity_stats[i] = {
            "n_params": count_total_parameters(layer),
            "n_zero_params": count_zero_parameters(layer),
            "attn_heads_n_params": count_total_parameters(layer.self_attn),
            "attn_heads_n_zero_params": count_zero_parameters(layer.self_attn),
            "ffn_n_params": count_total_parameters(layer.mlp),
            "ffn_n_zero_params": count_zero_parameters(layer.mlp),
        }
    # Check total sparsity
    sparsity_stats["total"] = {
        "n_params": count_total_parameters(model),
        "n_zero_params": count_zero_parameters(model),
        "attn_heads_n_params": sum(sparsity_stats[i]["attn_heads_n_params"] for i in sparsity_stats),
        "attn_heads_n_zero_params": sum(sparsity_stats[i]["attn_heads_n_zero_params"] for i in sparsity_stats),
        "ffn_n_params": sum(sparsity_stats[i]["ffn_n_params"] for i in sparsity_stats),
        "ffn_n_zero_params": sum(sparsity_stats[i]["ffn_n_zero_params"] for i in sparsity_stats),
    }

    if print_results:
        headers = [
            "Layer",
            "#Total\nParams",
            "#Total\n%Zero",
            "#Total\n%Pruned",
            "#Attn-Heads\nParams",
            "#Attn-Heads\n%Zero",
            "#Attn Heads\n%Pruned",
            "#FFN\nParams",
            "#FFN\n%Zero",
            "#FFN\n%Pruned",
        ]
        table = [
            [
                key,
                format_number(values["n_params"]),
                f"{values['n_zero_params']/values['n_params']*100:.2f}%" if values["n_params"] else "-",
                (
                    f"{(original_model_stats[key]['n_params']-values['n_params'])/original_model_stats[key]['n_params']*100:.2f}%"
                    if original_model_stats and original_model_stats[key]["n_params"]
                    else "-"
                ),
                format_number(values["attn_heads_n_params"]),
                (
                    f"{values['attn_heads_n_zero_params']/values['attn_heads_n_params']*100:.2f}%"
                    if values["attn_heads_n_params"]
                    else "-"
                ),
                (
                    f"{(original_model_stats[key]['attn_heads_n_params']-values['attn_heads_n_params'])/original_model_stats[key]['attn_heads_n_params']*100:.2f}%"
                    if original_model_stats and original_model_stats[key]["attn_heads_n_params"]
                    else "-"
                ),
                format_number(values["ffn_n_params"]),
                f"{values['ffn_n_zero_params']/values['ffn_n_params']*100:.2f}%" if values["ffn_n_params"] else "-",
                (
                    f"{(original_model_stats[key]['ffn_n_params']-values['ffn_n_params'])/original_model_stats[key]['ffn_n_params']*100:.2f}%"
                    if original_model_stats and original_model_stats[key]["ffn_n_params"]
                    else "-"
                ),
            ]
            for key, values in sparsity_stats.items()
        ]
        print(tabulate.tabulate(table, headers=headers, tablefmt="pretty", colalign=("left",)))
        sys.stdout.flush()

    # Calculate total flops, macs, params, zero_params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore FutureWarning on `truncation_strategy`

        total_flops, total_macs, total_params, total_zero_params = count_flops_macs_params(
            model, tokenizer, print_results=False
        )

    return (
        sparsity_stats,
        {
            "flops": total_flops,
            "macs": total_macs,
            "params": total_params,
            "zero_params": total_zero_params,
        },
    )


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
