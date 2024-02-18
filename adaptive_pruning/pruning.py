import warnings

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import prune_linear_layer


def prune_attention_heads(model: PreTrainedModel, heads_to_prune: dict[int, list[int]]) -> None:
    """
    Prune the specified attention heads in the model.
    Can remove all the attention heads in the specified layers.

    :param model: The transformers pytorch model to prune
    :param heads_to_prune: A dictionary with the layer indices as keys and a list of head indices to prune as values
    """
    with warnings.catch_warnings(category=UserWarning):
        warnings.simplefilter("ignore")  # ignore "no-op" warning when pruning all heads in a layer
        model.prune_heads(heads_to_prune)


def prune_attention_layers(model: PreTrainedModel, layers_to_prune: list[int]) -> None:
    """
    Prune the specified attention layers in the model.
    Simply delete all the attention heads in the specified layers.

    :param model: The transformers pytorch model to prune
    :param layers_to_prune: A list of layer indices to prune
    """
    heads_to_prune = {layer_index: list(range(model.config.num_attention_heads)) for layer_index in layers_to_prune}
    prune_attention_heads(model, heads_to_prune)
    # remove bias
    for layer_index in layers_to_prune:
        for name in ["query", "key", "value"]:
            param = model.encoder.layer[layer_index].attention.self.__getattr__(name)
            param.bias = None
        model.encoder.layer[layer_index].attention.output.dense.bias = None


def prune_ffn_neurons(model: PreTrainedModel, neurons_to_prune: dict[int, list[int]]) -> None:
    """
    Prune the specified feed forward neurons in the model.

    :param model: The transformers pytorch model to prune
    :param neurons_to_prune: A dictionary with the layer indices as keys and a list of neuron indices to prune as values
    """
    for layer_index, neurons in neurons_to_prune.items():
        neurons_indexes_to_keep = torch.LongTensor(list(set(range(model.config.intermediate_size)) - set(neurons)))

        model.encoder.layer[layer_index].intermediate.dense = prune_linear_layer(
            model.encoder.layer[layer_index].intermediate.dense,
            neurons_indexes_to_keep,
            dim=0,
        )
        model.encoder.layer[layer_index].output.dense = prune_linear_layer(
            model.encoder.layer[layer_index].output.dense,
            neurons_indexes_to_keep,
            dim=1,
        )


def prune_ffn_layers(model: PreTrainedModel, layers_to_prune: list[int]) -> None:
    """
    Prune the specified feed forward layers in the model.
    Simply replace the feed forward layers with a no-op n-0 and 0-m linear layers.

    :param model: The transformers pytorch model to prune
    :param layers_to_prune: A list of layer indices to prune
    """
    with warnings.catch_warnings(category=UserWarning):
        warnings.simplefilter("ignore")  # ignore "no-op" warning when pruning all neurons in a layer
        for layer_index in layers_to_prune:
            model.encoder.layer[layer_index].intermediate.dense = nn.Linear(
                model.encoder.layer[layer_index].intermediate.dense.in_features, 0
            )
            model.encoder.layer[layer_index].output.dense = nn.Linear(
                0, model.encoder.layer[layer_index].output.dense.out_features
            )
    # remove bias
    for layer_index in layers_to_prune:
        model.encoder.layer[layer_index].intermediate.dense.bias = None
        model.encoder.layer[layer_index].output.dense.bias = None


def _prune_embedding_layer(layer: nn.Embedding, index: torch.LongTensor, dim: int = 1) -> nn.Embedding:
    """Adapted from transformers.pytorch_utils.prune_linear_layer"""
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()

    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Embedding(new_size[0], new_size[1]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    return new_layer


def _prune_layer_norm(layer: nn.LayerNorm, index: torch.LongTensor) -> nn.LayerNorm:
    """Adapted from transformers.pytorch_utils.prune_linear_layer"""
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(0, index).clone().detach()
    if layer.bias is not None:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[0] = len(index)
    new_layer = nn.LayerNorm(new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def prune_hidden_state(model: PreTrainedModel, neurons_to_prune: list[int]) -> None:
    """
    Prune specific neurons from all hidden states along the model, including embeddings layer

    :param model: The transformers pytorch model to prune
    :param neurons_to_prune: List of neurons in dimensions to prune from the add hidden states along the model
    """
    neurons_indexes_to_keep = torch.LongTensor(list(set(range(model.config.hidden_size)) - set(neurons_to_prune)))

    # for name in ["word_embeddings", "position_embeddings", "token_type_embeddings"]:
    #     module = getattr(model.embeddings, name, None)
    #     if module is not None:
    #         module.weight = nn.Parameter(module.weight[neurons_indexes_to_keep])
    model.embeddings.word_embeddings = _prune_embedding_layer(
        model.embeddings.word_embeddings,
        neurons_indexes_to_keep,
    )
    model.embeddings.position_embeddings = _prune_embedding_layer(
        model.embeddings.position_embeddings,
        neurons_indexes_to_keep,
    )
    model.embeddings.token_type_embeddings = _prune_embedding_layer(
        model.embeddings.token_type_embeddings,
        neurons_indexes_to_keep,
    )
    model.embeddings.LayerNorm = _prune_layer_norm(
        model.embeddings.LayerNorm,
        neurons_indexes_to_keep,
    )

    for layer in model.encoder.layer:
        # attention in dimension
        layer.attention.self.query = prune_linear_layer(
            layer.attention.self.query,
            neurons_indexes_to_keep,
            dim=1,
        )
        layer.attention.self.key = prune_linear_layer(
            layer.attention.self.key,
            neurons_indexes_to_keep,
            dim=1,
        )
        layer.attention.self.value = prune_linear_layer(
            layer.attention.self.value,
            neurons_indexes_to_keep,
            dim=1,
        )
        # attention out dimension
        layer.attention.output.dense = prune_linear_layer(
            layer.attention.output.dense,
            neurons_indexes_to_keep,
            dim=0,
        )
        layer.attention.output.LayerNorm = _prune_layer_norm(
            layer.attention.output.LayerNorm,
            neurons_indexes_to_keep,
        )
        # ffn in dimension
        layer.intermediate.dense = prune_linear_layer(
            layer.intermediate.dense,
            neurons_indexes_to_keep,
            dim=1,
        )
        # ffn out dimension
        layer.output.dense = prune_linear_layer(
            layer.output.dense,
            neurons_indexes_to_keep,
            dim=0,
        )
        layer.output.LayerNorm = _prune_layer_norm(
            layer.output.LayerNorm,
            neurons_indexes_to_keep,
        )
    # pooler
    model.pooler.dense = prune_linear_layer(
        model.pooler.dense,
        neurons_indexes_to_keep,
        dim=1,
    )
    # update config
    model.config.hidden_size = len(neurons_indexes_to_keep)
