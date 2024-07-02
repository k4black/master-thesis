from __future__ import annotations

from typing import NamedTuple, Any
import sys

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from calflops import calculate_flops
import tabulate


def count_flops_macs_params(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 8,
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
        print_results=print_results,
        print_detailed=False,
        output_as_string=False,
        output_precision=4,
    )
    params = count_parameters(model, require_grad=None, print_results=False)
    if print_results:
        print(f"FLOPs:{flops}   MACs:{macs}   Params:{params} \n")
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
    
    
def measure_original_model_stats(model: PreTrainedModel) -> dict[str, Any]:
    base_model = model.base_model if hasattr(model, "base_model") else model
    assert 'llama' in base_model.config.model_type.lower(), f"Only llama models are supported, got {base_model.config.model_type}"
        
    model_stats = {
        'total': count_total_parameters(model),
        'base': count_total_parameters(base_model),
    }
    for i, layer in enumerate(base_model.layers):
        model_stats[f'layer_{i}'] = count_total_parameters(layer)
    
    return model_stats


def measure_pruned_model_stats(model: PreTrainedModel, original_model_stats: dict[str, Any] | None = None) -> dict[str, Any]:
    base_model = model.base_model if hasattr(model, "base_model") else model
    assert 'llama' in base_model.config.model_type.lower(), f"Only llama models are supported, got {base_model.config.model_type}"
         
    # Check total sparsity
    sparsity_stats = {
        'total': {
            'num_original_parameters': original_model_stats['total'] if original_model_stats else None,
            'num_parameters': count_total_parameters(model),
            'num_zero_parameters': count_zero_parameters(model),
            'num_nonzero_parameters': count_nonzero_parameters(model),
        },
        'base': {
            'num_original_parameters': original_model_stats['base'] if original_model_stats else None,
            'num_parameters': count_total_parameters(base_model),
            'num_zero_parameters': count_zero_parameters(base_model),
            'num_nonzero_parameters': count_nonzero_parameters(base_model),
        },
    }
    # Check sparsity by layer (unstructured and structured)
    for i, layer in enumerate(base_model.layers):
        sparsity_stats[f'layer_{i}'] = {
            'num_original_parameters': original_model_stats[f'layer_{i}'] if original_model_stats else None,
            'num_parameters': count_total_parameters(layer),
            'num_zero_parameters': count_zero_parameters(layer),
            'num_nonzero_parameters': count_nonzero_parameters(layer),
        }
    
    # Add percentage columns
    for key, stats in sparsity_stats.items():
        stats['percentage_original_pruned'] = (stats['num_original_parameters'] - stats['num_parameters']) / stats['num_original_parameters'] * 100 if stats['num_original_parameters'] else None
        stats['percentage_original_zero'] = stats['num_zero_parameters'] / stats['num_original_parameters'] * 100 if stats['num_original_parameters'] else None
        stats['percentage_original_pruned_or_zero'] = (stats['num_original_parameters'] - stats['num_parameters'] + stats['num_zero_parameters']) / stats['num_original_parameters'] * 100 if stats['num_original_parameters'] else None
        stats['percentage_zero'] = stats['num_zero_parameters'] / stats['num_parameters'] * 100

    return sparsity_stats
    

def print_measure_table(stats: dict[str, Any]) -> None:
    headers = ['Module', '#Params\n(Original)', '#Params\n(Pruned)', '#Params\n(Zero)', '#Params\n(Nonzero)', '%Pruned', '%Zero', '%Pruned or Zero', '%Zero\n(Current)']
    table: list[list[Any]] = []
    for key, values in stats.items():
        table.append([
            key,
            format_number(values['num_original_parameters']) if values['num_original_parameters'] else '-',
            format_number(values['num_parameters']),
            format_number(values['num_zero_parameters']),
            format_number(values['num_nonzero_parameters']),
            f"{values['percentage_original_pruned']:.2f}%" if values['percentage_original_pruned'] else '-',
            f"{values['percentage_original_zero']:.2f}%" if values['percentage_original_zero'] else '-',
            f"{values['percentage_original_pruned_or_zero']:.2f}%" if values['percentage_original_pruned_or_zero'] else '-',
            f"{values['percentage_zero']:.2f}%",
        ])
    print(tabulate.tabulate(table, headers=headers, tablefmt='pretty'))
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


def nullify_attention_heads(model: PreTrainedModel, heads_to_nullify: dict[int, list[int]]) -> None:
    """
    Nullify weights of the specified attention heads in the model.

    :param model: The transformers pytorch model to nullify
    :param heads_to_nullify: A dictionary with the layer indices as keys and a list of head indices to nullify as values
    """
    model, architecture = model.base_model, model.config.model_type
    head_size = model.config.hidden_size // model.config.num_attention_heads

    if architecture == "bert":
        for layer, heads in heads_to_nullify.items():
            for name in ["query", "key", "value"]:
                param = model.encoder.layer[layer].attention.self.__getattr__(name)
                for head_index in heads:
                    param.weight[head_index * head_size: (head_index + 1) * head_size] = 0
                    param.bias[head_index * head_size: (head_index + 1) * head_size] = 0

    elif architecture == "llama":
        num_heads_per_group = model.config.num_attention_heads // model.config.num_key_value_heads

        for layer, heads in heads_to_nullify.items():
            heads_grouped = list(set([i // num_heads_per_group for i in heads]))
            heads_not_grouped = [i * num_heads_per_group + j for i in heads_grouped for j in range(num_heads_per_group)]
            # q and out are full sized
            for head_index in heads_not_grouped:
                model.layers[layer].self_attn.q_proj.weight[head_index * head_size: (head_index + 1) * head_size] = 0
                model.layers[layer].self_attn.o_proj.weight[:, head_index * head_size: (head_index + 1) * head_size] = 0
            # k and v are grouped
            for head_index in heads_grouped:
                model.layers[layer].self_attn.k_proj.weight[head_index * head_size: (head_index + 1) * head_size] = 0
                model.layers[layer].self_attn.v_proj.weight[head_index * head_size: (head_index + 1) * head_size] = 0
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def nullify_attention_layers(model: PreTrainedModel, layers_to_nullify: list[int]) -> None:
    """
    Nullify the specified attention layers in the model.

    :param model: The transformers pytorch model to nullify
    :param layers_to_nullify: A list of layer indices to nullify
    """
    model, architecture = model.base_model, model.config.model_type
    heads_to_nullify = {layer_index: list(range(model.config.num_attention_heads)) for layer_index in layers_to_nullify}

    nullify_attention_heads(model, heads_to_nullify)

    # remove bias
    if architecture == "bert":
        for layer_index in layers_to_nullify:
            for name in ["query", "key", "value"]:
                param = model.encoder.layer[layer_index].attention.self.__getattr__(name)
                param.bias = None
            model.encoder.layer[layer_index].attention.output.dense.bias = None
    elif architecture == "llama":
        for layer_index in layers_to_nullify:
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                param = model.layers[layer_index].self_attn.__getattr__(name)
                param.bias = None
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def nullify_ffn_neurons(model: PreTrainedModel, neurons_to_nullify: dict[int, list[int]]) -> None:
    """
    Nullify the specified feed forward neurons in the model.

    :param model: The transformers pytorch model to nullify
    :param neurons_to_nullify: A dictionary with the layer indices as keys and a list of neuron indices to nullify as values
    """
    model, architecture = model.base_model, model.config.model_type

    if architecture == "bert":
        for layer_index, neurons in neurons_to_nullify.items():
            intermediate = model.encoder.layer[layer_index].intermediate.dense
            intermediate.weight[neurons, :] = 0
            intermediate.bias[neurons] = 0
            output = model.encoder.layer[layer_index].output.dense
            output.weight[:, neurons] = 0
    elif architecture == "llama":
        for layer_index, neurons in neurons_to_nullify.items():
            gate_proj = model.layers[layer_index].mlp.gate_proj
            gate_proj.weight[neurons, :] = 0
            if gate_proj.bias is not None:
                gate_proj.bias[neurons] = 0
            up_proj = model.layers[layer_index].mlp.up_proj
            up_proj.weight[neurons, :] = 0
            if up_proj.bias is not None:
                up_proj.bias[neurons] = 0
            down_proj = model.layers[layer_index].mlp.down_proj
            down_proj.weight[:, neurons] = 0
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def nullify_ffn_layers(model: PreTrainedModel, layers_to_nullify: list[int]) -> None:
    """
    Nullify the specified feed forward layers in the model.

    :param model: The transformers pytorch model to nullify
    :param layers_to_nullify: A list of layer indices to nullify
    """
    model, architecture = model.base_model, model.config.model_type
    neurons_to_nullify = {layer_index: list(range(model.config.intermediate_size)) for layer_index in layers_to_nullify}

    nullify_ffn_neurons(model, neurons_to_nullify)

    # remove bias
    if architecture == "bert":
        for layer_index in layers_to_nullify:
            intermediate = model.encoder.layer[layer_index].intermediate.dense
            output = model.encoder.layer[layer_index].output.dense
            intermediate.bias = None
            output.bias = None
    elif architecture == "llama":
        for layer_index in layers_to_nullify:
            gate_proj = model.layers[layer_index].mlp.gate_proj
            up_proj = model.layers[layer_index].mlp.up_proj
            down_proj = model.layers[layer_index].mlp.down_proj
            gate_proj.bias = None
            up_proj.bias = None
            down_proj.bias = None
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def _nullify_embedding_layer(layer: nn.Embedding, index: torch.LongTensor) -> None:
    index = index.to(layer.weight.device)
    layer.weight[:, index] = 0


def _nullify_layer_norm(layer: nn.LayerNorm, index: torch.LongTensor) -> None:
    index = index.to(layer.weight.device)
    layer.weight[index] = 0
    if layer.bias is not None:
        layer.bias[index] = 0


def nullify_hidden_state(model: PreTrainedModel, neurons_to_nullify: list[int]) -> None:
    """
    Nullify specific neurons from all hidden states along the model, including embeddings layer

    :param model: The transformers pytorch model to nullify
    :param neurons_to_nullify: List of neurons in dimensions to nullify from the add hidden states along the model
    """
    neurons_indexes_to_nullify = torch.LongTensor(list(set(neurons_to_nullify)))

    # for name in ["word_embeddings", "position_embeddings", "token_type_embeddings"]:
    #     module = getattr(model.embeddings, name, None)
    #     if module is not None:
    #         module.weight = nn.Parameter(module.weight[neurons_indexes_to_keep])
    _nullify_embedding_layer(
        model.embeddings.word_embeddings,
        neurons_indexes_to_nullify,
    )
    _nullify_embedding_layer(
        model.embeddings.position_embeddings,
        neurons_indexes_to_nullify,
    )
    _nullify_embedding_layer(
        model.embeddings.token_type_embeddings,
        neurons_indexes_to_nullify,
    )
    _nullify_layer_norm(
        model.embeddings.LayerNorm,
        neurons_indexes_to_nullify,
    )

    for layer in model.encoder.layer:
        raise NotImplementedError()


def tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.cpu().numpy().tolist()
