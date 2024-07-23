import warnings

import peft.tuners.lora
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.pytorch_utils import prune_linear_layer as transformers_prune_linear_layer

from adaptive_pruning.utils import get_base_model


def prune_attention_heads(model: PreTrainedModel, heads_to_prune: dict[int, list[int]]) -> None:
    """
    Prune the specified attention heads in the model.
    Can remove all the attention heads in the specified layers.

    :param model: The transformers pytorch model to prune
    :param heads_to_prune: A dictionary with the layer indices as keys and a list of head indices to prune as values
    """
    model, architecture = get_base_model(model), model.config.model_type

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore "no-op" warning when pruning all heads in a layer

        if architecture == "bert":
            # bert model has prune heads function
            model.prune_heads(heads_to_prune)

        elif architecture == "llama":
            num_heads = model.config.num_attention_heads
            num_grouped_heads = model.config.num_key_value_heads
            num_heads_per_group = num_heads // num_grouped_heads
            head_dim = model.config.hidden_size // model.config.num_attention_heads

            for layer_idx, heads in heads_to_prune.items():
                attention_layer = model.layers[layer_idx].self_attn

                # given head nums fill up to group_size to prune full groups
                #   e.g. having heads 8 heads with group_size 2:
                #   [1, 7] -> heads_grouped=[0, 3] (
                #   [1, 7] -> index_to_keep_grouped=[0...head_dim-1, 3*head_dim...4*head_dim-1]
                #   [1, 7] -> index_to_keep_full=index_to_keep_grouped repeat_interleave 2=[0, 1, 3, 4, 5, 6]
                heads_grouped = list(set([i // num_heads_per_group for i in heads]))
                index_to_prune_grouped = torch.LongTensor(
                    [i * head_dim + j for i in heads_grouped for j in range(head_dim)]
                )
                index_to_prune_full = torch.LongTensor(
                    [i * num_heads_per_group + j for i in index_to_prune_grouped for j in range(num_heads_per_group)]
                )

                # Prune values
                attention_layer.q_proj = _prune_linear_layer(
                    attention_layer.q_proj, index_to_prune_full, input_dim=False
                )
                attention_layer.k_proj = _prune_linear_layer(
                    attention_layer.k_proj, index_to_prune_grouped, input_dim=False
                )
                attention_layer.v_proj = _prune_linear_layer(
                    attention_layer.v_proj, index_to_prune_grouped, input_dim=False
                )
                attention_layer.o_proj = _prune_linear_layer(
                    attention_layer.o_proj, index_to_prune_full, input_dim=True
                )

                # Update hyper params and store pruned heads
                attention_layer.num_heads = attention_layer.num_heads - (len(heads_grouped) * num_heads_per_group)
                attention_layer.num_key_value_heads = attention_layer.num_key_value_heads - len(heads_grouped)
                # attention_layer.num_key_value_groups = attention_layer.num_key_value_groups  # keep the same group size
                # attention_layer.hidden_size = attention_layer.hidden_size - (len(heads_grouped) * num_heads_per_group * head_dim)


def prune_attention_layers(model: PreTrainedModel, layers_to_prune: list[int]) -> None:
    """
    Prune the specified attention layers in the model.
    Simply delete all the attention heads in the specified layers.

    :param model: The transformers pytorch model to prune
    :param layers_to_prune: A list of layer indices to prune
    """
    model, architecture = get_base_model(model), model.config.model_type

    heads_to_prune = {layer_index: list(range(model.config.num_attention_heads)) for layer_index in layers_to_prune}
    prune_attention_heads(model, heads_to_prune)

    # remove bias (output projection mostly)
    if architecture == "bert":
        for layer_index in heads_to_prune.keys():
            for name in ["query", "key", "value"]:
                param = model.encoder.layer[layer_index].attention.self.__getattr__(name)
                param.bias = None
            model.encoder.layer[layer_index].attention.output.dense.bias = None

    elif architecture == "llama":
        for layer_index in heads_to_prune.keys():
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                param = model.layers[layer_index].self_attn.__getattr__(name)
                param.bias = None


def prune_ffn_neurons(model: PreTrainedModel, neurons_to_prune: dict[int, list[int]]) -> None:
    """
    Prune the specified feed forward neurons in the model.

    :param model: The transformers pytorch model to prune
    :param neurons_to_prune: A dictionary with the layer indices as keys and a list of neuron indices to prune as values
    """
    model, architecture = get_base_model(model), model.config.model_type

    for layer_index, neurons in neurons_to_prune.items():
        neurons_indexes_to_prune = torch.LongTensor(neurons)

        if architecture == "bert":
            layer = model.encoder.layer[layer_index]

            layer.intermediate.dense = _prune_linear_layer(
                layer.intermediate.dense,
                neurons_indexes_to_prune,
                input_dim=False,
            )
            layer.output.dense = _prune_linear_layer(
                layer.output.dense,
                neurons_indexes_to_prune,
                input_dim=True,
            )
        elif architecture == "llama":
            layer = model.layers[layer_index].mlp

            layer.gate_proj = _prune_linear_layer(
                layer.gate_proj,
                neurons_indexes_to_prune,
                input_dim=False,
            )
            layer.up_proj = _prune_linear_layer(
                layer.up_proj,
                neurons_indexes_to_prune,
                input_dim=False,
            )
            layer.down_proj = _prune_linear_layer(
                layer.down_proj,
                neurons_indexes_to_prune,
                input_dim=True,
            )
            layer.intermediate_size = layer.intermediate_size - len(neurons)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")


def prune_ffn_layers(model: PreTrainedModel, layers_to_prune: list[int]) -> None:
    """
    Prune the specified feed forward layers in the model.
    Simply replace the feed forward layers with a no-op n-0 and 0-m linear layers.

    :param model: The transformers pytorch model to prune
    :param layers_to_prune: A list of layer indices to prune
    """
    model, architecture = get_base_model(model), model.config.model_type

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore "no-op" warning when pruning all neurons in a layer

        if architecture == "bert":
            for layer_index in layers_to_prune:
                model.encoder.layer[layer_index].intermediate.dense = nn.Linear(
                    model.encoder.layer[layer_index].intermediate.dense.in_features,
                    0,
                    device=model.encoder.layer[layer_index].intermediate.dense.weight.device,
                )
                model.encoder.layer[layer_index].output.dense = nn.Linear(
                    0,
                    model.encoder.layer[layer_index].output.dense.out_features,
                    device=model.encoder.layer[layer_index].output.dense.weight.device,
                )
            # remove bias
            for layer_index in layers_to_prune:
                model.encoder.layer[layer_index].intermediate.dense.bias = None
                model.encoder.layer[layer_index].output.dense.bias = None
        elif architecture == "llama":
            for layer_index in layers_to_prune:
                model.layers[layer_index].mlp.gate_proj = nn.Linear(
                    model.layers[layer_index].mlp.gate_proj.in_features,
                    0,
                    device=model.layers[layer_index].mlp.gate_proj.weight.device,
                )
                model.layers[layer_index].mlp.up_proj = nn.Linear(
                    model.layers[layer_index].mlp.up_proj.in_features,
                    0,
                    device=model.layers[layer_index].mlp.up_proj.weight.device,
                )
                model.layers[layer_index].mlp.down_proj = nn.Linear(
                    0,
                    model.layers[layer_index].mlp.down_proj.out_features,
                    device=model.layers[layer_index].mlp.down_proj.weight.device,
                )
            # remove bias
            for layer_index in layers_to_prune:
                model.layers[layer_index].mlp.gate_proj.bias = None
                model.layers[layer_index].mlp.up_proj.bias = None
                model.layers[layer_index].mlp.down_proj.bias = None
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")


def _prune_linear_layer(layer: nn.Linear, index_to_prune: torch.LongTensor, input_dim: bool = False) -> nn.Linear:
    """Just run the prune_linear_layer function from transformers.pytorch_utils"""
    dim = 1 if input_dim else 0
    index_to_keep = torch.LongTensor(list(set(range(layer.weight.size(dim))) - {int(i) for i in index_to_prune}))
    if isinstance(layer, nn.Linear):
        return transformers_prune_linear_layer(layer, index_to_keep, dim=dim)
    elif isinstance(layer, peft.tuners.lora.Linear):
        raise NotImplementedError("Pruning LORA layers is not supported yet")
        # TODO: prune with LORA weights, for now only after merging
        # https://huggingface.co/docs/peft/main/en/developer_guides/lora#merge-lora-weights-into-the-base-model
        layer: peft.tuners.lora.Linear
        layer.base_layer = transformers_prune_linear_layer(layer.base_layer, index_to_keep, dim=dim)
        # if input_dim:
        # layer.lora_A
        return layer
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")


def _prune_embedding_layer_hidden_states(layer: nn.Embedding, index_to_prune: torch.LongTensor) -> nn.Embedding:
    """Adapted from transformers.pytorch_utils.prune_linear_layer"""
    index_to_keep = torch.LongTensor(list(set(range(layer.embedding_dim)) - {int(i) for i in index_to_prune}))

    index_to_keep = index_to_keep.to(layer.weight.device)
    W = layer.weight.index_select(1, index_to_keep).clone().detach()

    new_layer = nn.Embedding(num_embeddings=layer.num_embeddings, embedding_dim=len(index_to_keep)).to(
        layer.weight.device
    )
    _original_requires_grad = layer.weight.requires_grad
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = _original_requires_grad
    return new_layer


def _prune_layer_norm(layer: nn.LayerNorm, index_to_prune: torch.LongTensor) -> nn.LayerNorm:
    """Adapted from transformers.pytorch_utils.prune_linear_layer"""
    index_to_keep = torch.LongTensor(list(set(range(layer.weight.size(0))) - {int(i) for i in index_to_prune}))

    index_to_keep = index_to_keep.to(layer.weight.device)
    W = layer.weight.index_select(0, index_to_keep).clone().detach()
    if hasattr(layer, "bias") and layer.bias is not None:
        b = layer.bias[index_to_keep].clone().detach()
    new_size = list(layer.weight.size())
    new_size[0] = len(index_to_keep)
    new_layer = nn.LayerNorm(new_size[0], bias=hasattr(layer, "bias") and layer.bias is not None).to(
        layer.weight.device
    )
    _original_requires_grad = layer.weight.requires_grad
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = _original_requires_grad
    if hasattr(layer, "bias") and layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = _original_requires_grad
    return new_layer


def prune_hidden_states(model: PreTrainedModel, hidden_states_to_prune: list[int]) -> None:
    """
    Prune specific neurons from all hidden states along the model, including embeddings layer

    :param model: The transformers pytorch model to prune
    :param hidden_states_to_prune: List of neurons in dimensions to prune from the add hidden states along the model
    """
    print(model)
    base_model, architecture = get_base_model(model), model.config.model_type
    hidden_states_to_prune = torch.LongTensor(hidden_states_to_prune)

    if architecture == "bert":
        if hasattr(model, "cls"):
            if isinstance(model.cls, BertOnlyMLMHead):
                model.cls.predictions.transform.dense = _prune_linear_layer(
                    model.cls.predictions.transform.dense,
                    hidden_states_to_prune,
                    input_dim=True,
                )
                model.cls.predictions.transform.dense = _prune_linear_layer(
                    model.cls.predictions.transform.dense,
                    hidden_states_to_prune,
                    input_dim=False,
                )
                model.cls.predictions.transform.LayerNorm = _prune_layer_norm(
                    model.cls.predictions.transform.LayerNorm,
                    hidden_states_to_prune,
                )
                model.cls.predictions.decoder = _prune_linear_layer(
                    model.cls.predictions.decoder,
                    hidden_states_to_prune,
                    input_dim=True,
                )
        base_model.embeddings.word_embeddings = _prune_embedding_layer_hidden_states(
            base_model.embeddings.word_embeddings,
            hidden_states_to_prune,
        )
        base_model.embeddings.position_embeddings = _prune_embedding_layer_hidden_states(
            base_model.embeddings.position_embeddings,
            hidden_states_to_prune,
        )
        base_model.embeddings.token_type_embeddings = _prune_embedding_layer_hidden_states(
            base_model.embeddings.token_type_embeddings,
            hidden_states_to_prune,
        )
        base_model.embeddings.LayerNorm = _prune_layer_norm(
            base_model.embeddings.LayerNorm,
            hidden_states_to_prune,
        )

        for layer in base_model.encoder.layer:
            # attention in dimension
            layer.attention.self.query = _prune_linear_layer(
                layer.attention.self.query,
                hidden_states_to_prune,
                input_dim=True,
            )
            layer.attention.self.key = _prune_linear_layer(
                layer.attention.self.key,
                hidden_states_to_prune,
                input_dim=True,
            )
            layer.attention.self.value = _prune_linear_layer(
                layer.attention.self.value,
                hidden_states_to_prune,
                input_dim=True,
            )
            # attention out dimension
            layer.attention.output.dense = _prune_linear_layer(
                layer.attention.output.dense,
                hidden_states_to_prune,
                input_dim=False,
            )
            layer.attention.output.LayerNorm = _prune_layer_norm(
                layer.attention.output.LayerNorm,
                hidden_states_to_prune,
            )
            # ffn dimension
            layer.intermediate.dense = _prune_linear_layer(
                layer.intermediate.dense,
                hidden_states_to_prune,
                input_dim=True,
            )
            layer.output.dense = _prune_linear_layer(
                layer.output.dense,
                hidden_states_to_prune,
                input_dim=False,
            )
            layer.output.LayerNorm = _prune_layer_norm(
                layer.output.LayerNorm,
                hidden_states_to_prune,
            )
        # pooler
        if hasattr(base_model, "pooler") and base_model.pooler is not None:
            base_model.pooler.dense = _prune_linear_layer(
                base_model.pooler.dense,
                hidden_states_to_prune,
                input_dim=False,
            )
            base_model.pooler.dense = _prune_linear_layer(
                base_model.pooler.dense,
                hidden_states_to_prune,
                input_dim=True,
            )
    elif architecture == "llama":
        if hasattr(model, "lm_head"):
            model.lm_head = _prune_linear_layer(
                model.lm_head,
                hidden_states_to_prune,
                input_dim=True,
            )
        base_model.embed_tokens = _prune_embedding_layer_hidden_states(
            base_model.embed_tokens,
            hidden_states_to_prune,
        )
        base_model.norm = _prune_layer_norm(
            base_model.norm,
            hidden_states_to_prune,
        )

        for layer in base_model.layers:
            # layer norms
            layer.input_layernorm = _prune_layer_norm(
                layer.input_layernorm,
                hidden_states_to_prune,
            )
            layer.post_attention_layernorm = _prune_layer_norm(
                layer.post_attention_layernorm,
                hidden_states_to_prune,
            )

            # nullify attention layers hidden state
            layer.self_attn.q_proj = _prune_linear_layer(
                layer.self_attn.q_proj,
                hidden_states_to_prune,
                input_dim=True,
            )
            layer.self_attn.k_proj = _prune_linear_layer(
                layer.self_attn.k_proj,
                hidden_states_to_prune,
                input_dim=True,
            )
            layer.self_attn.v_proj = _prune_linear_layer(
                layer.self_attn.v_proj,
                hidden_states_to_prune,
                input_dim=True,
            )
            layer.self_attn.o_proj = _prune_linear_layer(
                layer.self_attn.o_proj,
                hidden_states_to_prune,
                input_dim=False,
            )

            # nullify ffn layers hidden state
            layer.mlp.gate_proj = _prune_linear_layer(
                layer.mlp.gate_proj,
                hidden_states_to_prune,
                input_dim=True,
            )
            layer.mlp.up_proj = _prune_linear_layer(
                layer.mlp.up_proj,
                hidden_states_to_prune,
                input_dim=True,
            )
            layer.mlp.down_proj = _prune_linear_layer(
                layer.mlp.down_proj,
                hidden_states_to_prune,
                input_dim=False,
            )

    # update config
    model.config.hidden_size = model.config.hidden_size - len(hidden_states_to_prune)
