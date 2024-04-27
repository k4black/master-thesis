from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from calflops import calculate_flops


class TargetModules(NamedTuple):
    attention_heads: str | list[str] | None
    attention_layers: str | list[str] | None
    ffn_neurons: str | list[str] | None
    ffn_layers: str | list[str] | None
    hidden_states: str | list[str] | None


ARCHITECTURE_TO_TARGET_MODULES = {
    "bert": TargetModules(
        attention_heads="attention.self",
        attention_layers="attention.output",
        ffn_neurons="output",
        ffn_layers="output.dense",
        hidden_states=["embeddings"],
    ),
}


def get_model_part_by_name(model: PreTrainedModel, name: str) -> nn.Module:
    pass



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


def count_parameters(
        model: nn.Module,
        require_grad: bool | None = None,
        *,
        print_results: bool = True,
) -> int:
    """
    Count the number of parameters in the model.
    :param model: torch.nn.Module or transformers.PreTrainedModel
    :param require_grad: if defined (not None), count only parameters that require/not-require grad
    :return: The number of parameters in the model
    """
    if require_grad is None:
        num_params = sum(p.numel() for p in model.parameters())
    else:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad == require_grad)

    if print_results:
        params_type = 'All' if require_grad is None else 'Trainable' if require_grad else 'Frozen'
        print(f"Number of {params_type} parameters: {num_params}")

    return num_params


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
    head_size = model.config.hidden_size // model.config.num_attention_heads
    
    for layer, heads in heads_to_nullify.items():
        for name in ["query", "key", "value"]:
            param = model.encoder.layer[layer].attention.self.__getattr__(name)
            for head_index in heads:
                param.weight[head_index * head_size: (head_index + 1) * head_size] = 0
                param.bias[head_index * head_size: (head_index + 1) * head_size] = 0


def nullify_attention_layers(model: PreTrainedModel, layers_to_nullify: list[int]) -> None:
    """
    Nullify the specified attention layers in the model.

    :param model: The transformers pytorch model to nullify
    :param layers_to_nullify: A list of layer indices to nullify
    """
    heads_to_nullify = {layer_index: list(range(model.config.num_attention_heads)) for layer_index in layers_to_nullify}
    nullify_attention_heads(model, heads_to_nullify)
    # remove bias
    for layer_index in layers_to_nullify:
        for name in ["query", "key", "value"]:
            param = model.encoder.layer[layer_index].attention.self.__getattr__(name)
            param.bias = None
        model.encoder.layer[layer_index].attention.output.dense.bias = None


def nullify_ffn_neurons(model: PreTrainedModel, neurons_to_nullify: dict[int, list[int]]) -> None:
    """
    Nullify the specified feed forward neurons in the model.

    :param model: The transformers pytorch model to nullify
    :param neurons_to_nullify: A dictionary with the layer indices as keys and a list of neuron indices to nullify as values
    """
    for layer_index, neurons in neurons_to_nullify.items():
        intermediate = model.encoder.layer[layer_index].intermediate.dense
        intermediate.weight[neurons, :] = 0
        intermediate.bias[neurons] = 0
        output = model.encoder.layer[layer_index].output.dense
        output.weight[:, neurons] = 0


def nullify_ffn_layers(model: PreTrainedModel, layers_to_nullify: list[int]) -> None:
    """
    Nullify the specified feed forward layers in the model.

    :param model: The transformers pytorch model to nullify
    :param layers_to_nullify: A list of layer indices to nullify
    """
    neurons_to_nullify = {layer_index: list(range(model.config.intermediate_size)) for layer_index in layers_to_nullify}
    nullify_ffn_neurons(model, neurons_to_nullify)
    # remove bias
    for layer_index in layers_to_nullify:
        intermediate = model.encoder.layer[layer_index].intermediate.dense
        output = model.encoder.layer[layer_index].output.dense
        intermediate.bias = None
        output.bias = None


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
