import warnings

import torch
import torch.nn as nn
from transformers import PreTrainedModel


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
