from __future__ import annotations

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from adaptive_pruning.utils import get_base_model


def nullify_attention_heads(model: PreTrainedModel, heads_to_nullify: dict[int, list[int]]) -> None:
    """
    Nullify weights of the specified attention heads in the model.

    :param model: The transformers pytorch model to nullify
    :param heads_to_nullify: A dictionary with the layer indices as keys and a list of head indices to nullify as values
    """
    model, architecture = get_base_model(model), model.config.model_type
    head_size = model.config.hidden_size // model.config.num_attention_heads

    if architecture == "bert":
        for layer, heads in heads_to_nullify.items():
            for name in ["query", "key", "value"]:
                param = model.encoder.layer[layer].attention.self.__getattr__(name)
                for head_index in heads:
                    _nullify_linear_layer(
                        param,
                        torch.arange(head_index * head_size, (head_index + 1) * head_size, dtype=torch.long),
                        input_dim=False,
                    )

    elif architecture == "llama":
        num_heads_per_group = model.config.num_attention_heads // model.config.num_key_value_heads

        for layer, heads in heads_to_nullify.items():
            heads_grouped = list(set([i // num_heads_per_group for i in heads]))
            heads_not_grouped = [i * num_heads_per_group + j for i in heads_grouped for j in range(num_heads_per_group)]
            # q and out are full sized
            for head_index in heads_not_grouped:
                _nullify_linear_layer(
                    model.layers[layer].self_attn.q_proj,
                    torch.arange(head_index * head_size, (head_index + 1) * head_size, dtype=torch.long),
                    input_dim=False,
                )
                _nullify_linear_layer(
                    model.layers[layer].self_attn.o_proj,
                    torch.arange(head_index * head_size, (head_index + 1) * head_size, dtype=torch.long),
                    input_dim=True,
                )
            # k and v are grouped
            for head_index in heads_grouped:
                _nullify_linear_layer(
                    model.layers[layer].self_attn.k_proj,
                    torch.arange(head_index * head_size, (head_index + 1) * head_size, dtype=torch.long),
                    input_dim=False,
                )
                _nullify_linear_layer(
                    model.layers[layer].self_attn.v_proj,
                    torch.arange(head_index * head_size, (head_index + 1) * head_size, dtype=torch.long),
                    input_dim=False,
                )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def nullify_attention_layers(model: PreTrainedModel, layers_to_nullify: list[int]) -> None:
    """
    Nullify the specified attention layers in the model.

    :param model: The transformers pytorch model to nullify
    :param layers_to_nullify: A list of layer indices to nullify
    """
    model, architecture = get_base_model(model), model.config.model_type
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
    model, architecture = get_base_model(model), model.config.model_type

    if architecture == "bert":
        for layer_index, neurons in neurons_to_nullify.items():
            _nullify_linear_layer(
                model.encoder.layer[layer_index].intermediate.dense,
                torch.tensor(neurons, dtype=torch.long),
                input_dim=False,
            )
            _nullify_linear_layer(
                model.encoder.layer[layer_index].output.dense,
                torch.tensor(neurons, dtype=torch.long),
                input_dim=True,
            )
    elif architecture == "llama":
        for layer_index, neurons in neurons_to_nullify.items():
            _nullify_linear_layer(
                model.layers[layer_index].mlp.gate_proj,
                torch.tensor(neurons, dtype=torch.long),
                input_dim=False,
            )
            _nullify_linear_layer(
                model.layers[layer_index].mlp.up_proj,
                torch.tensor(neurons, dtype=torch.long),
                input_dim=False,
            )
            _nullify_linear_layer(
                model.layers[layer_index].mlp.down_proj,
                torch.tensor(neurons, dtype=torch.long),
                input_dim=True,
            )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def nullify_ffn_layers(model: PreTrainedModel, layers_to_nullify: list[int]) -> None:
    """
    Nullify the specified feed forward layers in the model.

    :param model: The transformers pytorch model to nullify
    :param layers_to_nullify: A list of layer indices to nullify
    """
    model, architecture = get_base_model(model), model.config.model_type
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


def _nullify_embedding_layer_hidden_states(layer: nn.Embedding, index_to_nullify: torch.LongTensor) -> None:
    index_to_nullify = index_to_nullify.to(layer.weight.device)
    _original_requires_grad = layer.weight.requires_grad
    layer.weight.requires_grad = False
    layer.weight[:, index_to_nullify] = 0
    layer.weight.requires_grad = _original_requires_grad


def _nullify_layer_norm(layer: nn.LayerNorm, index_to_nullify: torch.LongTensor) -> None:
    index_to_nullify = index_to_nullify.to(layer.weight.device)
    _original_requires_grad = layer.weight.requires_grad
    layer.weight.requires_grad = False
    layer.weight[index_to_nullify] = 0
    layer.weight.requires_grad = _original_requires_grad
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.requires_grad = False
        layer.bias[index_to_nullify] = 0
        layer.weight.requires_grad = _original_requires_grad


def _nullify_linear_layer(layer: nn.Linear, index_to_nullify: torch.LongTensor, input_dim: bool = False) -> None:
    """
    Nullify the specified neurons in the linear layer.
    @see prune_linear_layer of transformers

    :param layer: nn.Linear layer to nullify inplace
    :param index_to_nullify: indices of neurons to nullify
    :param input_dim: if True, nullify input neurons, otherwise nullify output neurons
        Note: for linear layer, dim=0 in the matrix is the output neurons, dim=1 is the input neurons
    :return: None
    """
    _original_requires_grad = layer.weight.requires_grad
    index_to_nullify = index_to_nullify.to(layer.weight.device)
    if input_dim:
        layer.weight.requires_grad = False
        layer.weight[:, index_to_nullify] = 0
        layer.weight.requires_grad = _original_requires_grad
    else:
        layer.weight.requires_grad = False
        layer.weight[index_to_nullify, :] = 0
        layer.weight.requires_grad = _original_requires_grad
        if layer.bias is not None:
            layer.bias.requires_grad = False
            layer.bias[index_to_nullify] = 0
            layer.bias.requires_grad = _original_requires_grad


def nullify_hidden_states(model: PreTrainedModel, neurons_to_nullify: list[int]) -> None:
    """
    Nullify specific neurons from all hidden states along the model, including embeddings layer

    :param model: The transformers pytorch model to nullify
    :param neurons_to_nullify: List of neurons in dimensions to nullify from the add hidden states along the model
    """
    base_model, architecture = get_base_model(model), model.config.model_type
    hidden_states_to_nullify = torch.LongTensor(list(set(neurons_to_nullify)))

    if architecture == "bert":
        raise NotImplementedError("Not implemented for BERT")
    elif architecture == "llama":
        if hasattr(model, "lm_head"):
            _nullify_linear_layer(
                model.lm_head,
                hidden_states_to_nullify,
                input_dim=True,
            )
        _nullify_embedding_layer_hidden_states(
            base_model.embed_tokens,
            hidden_states_to_nullify,
        )
        _nullify_layer_norm(
            base_model.norm,
            hidden_states_to_nullify,
        )

        for layer in base_model.layers:
            # layer norms
            _nullify_layer_norm(
                layer.input_layernorm,
                hidden_states_to_nullify,
            )
            _nullify_layer_norm(
                layer.post_attention_layernorm,
                hidden_states_to_nullify,
            )

            # nullify attention layers hidden state
            _nullify_linear_layer(
                layer.self_attn.q_proj,
                hidden_states_to_nullify,
                input_dim=True,
            )
            _nullify_linear_layer(
                layer.self_attn.k_proj,
                hidden_states_to_nullify,
                input_dim=True,
            )
            _nullify_linear_layer(
                layer.self_attn.v_proj,
                hidden_states_to_nullify,
                input_dim=True,
            )
            _nullify_linear_layer(
                layer.self_attn.o_proj,
                hidden_states_to_nullify,
                input_dim=False,
            )

            # nullify ffn layers hidden state
            _nullify_linear_layer(
                layer.mlp.gate_proj,
                hidden_states_to_nullify,
                input_dim=True,
            )
            _nullify_linear_layer(
                layer.mlp.up_proj,
                hidden_states_to_nullify,
                input_dim=True,
            )
            _nullify_linear_layer(
                layer.mlp.down_proj,
                hidden_states_to_nullify,
                input_dim=False,
            )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
