from typing import Any

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel


def _register_pre_mask(module: nn.Module, mask: torch.Tensor, extra_mask: torch.Tensor | None) -> RemovableHandle:
    # Note: It is important to have separate function to limit shadowing of the inputs variable, e.g. layer

    def hook(_module: nn.Module, inputs: Any) -> Any:
        if isinstance(inputs, tuple):
            return inputs[0] * mask * (extra_mask or 1), *inputs[1:]
        else:
            return inputs * mask * (extra_mask or 1)

    handle = module.register_forward_pre_hook(hook)
    return handle


def _register_post_mask(module: nn.Module, mask: torch.Tensor, extra_mask: torch.Tensor | None) -> RemovableHandle:
    # Note: It is important to have separate function to limit shadowing of the inputs variable, e.g. layer

    def hook(_module: nn.Module, _inputs: Any, outputs: Any) -> Any:
        if isinstance(outputs, tuple):
            return outputs[0] * mask * (extra_mask or 1), *outputs[1:]
        else:
            return outputs * mask * (extra_mask or 1)

    handle = module.register_forward_hook(hook)
    return handle


def inject_attention_head_mask(
         model: PreTrainedModel, head_mask: torch.Tensor, extra_mask: torch.Tensor | None = None
) -> list[RemovableHandle]:
    """
    Inject mask into the model's attention heads such a way it emulates the effect of pruning.

    The mask is applied to the input of the attention heads, effectively zeroing out the input to the masked heads.

    For each layer:
        - self [hidden_size -> hidden_size]
        > mask [num_attention_heads] (broadcasted to [hidden_size])
        - output [hidden_size -> hidden_size]

    :param model: The transformers pytorch model to inject the mask into
    :param head_mask: The attention head mask to inject of shape [num_hidden_layers, num_attention_heads]
    """
    removable_handles = []
    attention_head_size = model.config.hidden_size // model.config.num_attention_heads

    for layer in range(model.config.num_hidden_layers):
        # TODO: check gradient is calc correctly
        # head_mask of shape [num_attention_heads] extend to have shape of [num_attention_heads*attention_head_size]
        broadcased_head_mask = head_mask[layer].repeat_interleave(attention_head_size)
        removable_handles.append(
            _register_post_mask(model.encoder.layer[layer].attention.self, broadcased_head_mask, extra_mask=extra_mask)
        )

    return removable_handles


def inject_attention_layer_mask(
        model: PreTrainedModel, layer_mask: torch.Tensor, extra_mask: torch.Tensor | None = None
) -> list[RemovableHandle]:
    """
    Inject mask into the model's attention layers such a way it emulates the effect of pruning.

    The mask is applied to the output of the attention layers, effectively zeroing out the output of the masked layers.

    For each layer:
        - self [hidden_size -> hidden_size]
        > mask [1]
        - output [hidden_size -> hidden_size]

    :param model: The transformers pytorch model to inject the mask into
    :param layer_mask: The attention layer mask to inject of shape [num_hidden_layers]
    """
    removable_handles = []

    for layer in range(model.config.num_hidden_layers):
        removable_handles.append(
            _register_pre_mask(model.encoder.layer[layer].attention.output, layer_mask[layer], extra_mask=extra_mask)
        )
        # to remove bias
        removable_handles.append(
            _register_post_mask(model.encoder.layer[layer].attention.output.dense, layer_mask[layer], extra_mask=extra_mask)
        )

    return removable_handles


def inject_ffn_neuron_mask(
        model: PreTrainedModel, neuron_mask: torch.Tensor, extra_mask: torch.Tensor | None = None
) -> list[RemovableHandle]:
    """
    Inject mask into the model's feed forward neurons such a way it emulates the effect of pruning.

    The mask is applied to the output of the feed forward neurons, effectively zeroing out the output of the masked neurons.

    For each layer:
        - intermediate [hidden_size -> intermediate_size]
        > mask [intermediate_size]
        - output [intermediate_size -> hidden_size]

    :param model: The transformers pytorch model to inject the mask into
    :param neuron_mask: The feed forward neuron mask to inject of shape [num_hidden_layers, intermediate_size]
    """
    removable_handles = []

    for layer in range(model.config.num_hidden_layers):
        removable_handles.append(
            _register_pre_mask(model.encoder.layer[layer].output, neuron_mask[layer], extra_mask=extra_mask)
        )

    return removable_handles


def inject_ffn_layer_mask(
        model: PreTrainedModel, layer_mask: torch.Tensor, extra_mask: torch.Tensor | None = None
) -> list[RemovableHandle]:
    """
    Inject mask into the model's feed forward layers such a way it emulates the effect of pruning.

    The mask is applied to the output of the feed forward layers, effectively zeroing out the output of the masked layers.

    For each layer:
        - intermediate [hidden_size -> intermediate_size]
        > mask [1]
        - output [intermediate_size -> hidden_size]

    :param model: The transformers pytorch model to inject the mask into
    :param layer_mask: The feed forward layer mask to inject of shape [num_hidden_layers]
    """
    removable_handles = []

    for layer in range(model.config.num_hidden_layers):
        removable_handles.append(
            _register_pre_mask(model.encoder.layer[layer].output, layer_mask[layer], extra_mask=extra_mask)
        )
        # to remove bias
        removable_handles.append(
            _register_post_mask(model.encoder.layer[layer].output.dense, layer_mask[layer], extra_mask=extra_mask)
        )

    return removable_handles


def inject_hidden_state_mask(
        model: PreTrainedModel, hidden_state_mask: torch.Tensor, extra_mask: torch.Tensor | None = None
) -> list[RemovableHandle]:
    """
    Inject mask into the model's hidden states such a way it emulates the effect of pruning.
    All hidden_size are subject to the same mask, as we have residual connections.

    The mask is applied to the hidden states, effectively zeroing out the hidden states of the masked layers.

    For each layer with hidden_size e.g. ffn, embeddings, encoder, etc:
        - output [hidden_size -> hidden_size]
        > mask [hidden_size]
        - output [hidden_size -> hidden_size]

    :param model: The transformers pytorch model to inject the mask into
    :param hidden_state_mask: The hidden state mask to inject of shape [hidden_size]
    """
    removable_handles = []

    # embeddings
    for name in ["word_embeddings", "position_embeddings", "token_type_embeddings"]:
        removable_handles.append(_register_post_mask(model.embeddings.__getattr__(name), hidden_state_mask))
        removable_handles.append(_register_pre_mask(model.embeddings.LayerNorm, hidden_state_mask))

    # encoder
    for layer in range(model.config.num_hidden_layers):
        # attentions
        removable_handles.append(_register_pre_mask(model.encoder.layer[layer].attention.output, hidden_state_mask))

        # ffn
        removable_handles.append(_register_pre_mask(model.encoder.layer[layer].intermediate, hidden_state_mask))
        removable_handles.append(_register_post_mask(model.encoder.layer[layer].output, hidden_state_mask))

    # pooler
    removable_handles.append(_register_pre_mask(model.pooler, hidden_state_mask))

    return removable_handles
