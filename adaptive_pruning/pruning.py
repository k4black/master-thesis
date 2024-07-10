import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.pytorch_utils import prune_linear_layer


def llama_prune_out_channels(
    layer: nn.Module, heads_indexes: Sequence[int], num_heads: int, head_dim: int
) -> nn.Module:
    # print("Prune IDX in HFAttentionPruner: ", idxs)
    # convert head indexes to neuron indexes e.g [0] -> [0, .., head_dim-1]
    idxs = [i * layer.head_dim + j for i in heads_indexes for j in range(head_dim)]
    assert len(idxs) % num_heads == 0
    for sub_layer in [layer.o_proj]:
        keep_idxs = list(set(range(sub_layer.out_features)) - set(idxs))
        keep_idxs.sort()
        sub_layer.out_features = sub_layer.out_features - len(idxs)

        sub_layer.weight = torch.nn.Parameter(sub_layer.weight.data[keep_idxs])
        if sub_layer.bias is not None:
            sub_layer.bias = torch.nn.Parameter(sub_layer.bias.data[keep_idxs])

    for sub_layer in [layer.q_proj, layer.k_proj, layer.v_proj]:
        keep_idxs = list(set(range(sub_layer.in_features)) - set(idxs))
        keep_idxs.sort()
        sub_layer.in_features = sub_layer.in_features - len(idxs)
        sub_layer.weight = torch.nn.Parameter(sub_layer.weight.data[:, keep_idxs])

    return layer


def prune_attention_heads(model: PreTrainedModel, heads_to_prune: dict[int, list[int]]) -> None:
    """
    Prune the specified attention heads in the model.
    Can remove all the attention heads in the specified layers.

    :param model: The transformers pytorch model to prune
    :param heads_to_prune: A dictionary with the layer indices as keys and a list of head indices to prune as values
    """
    model, architecture = model.base_model, model.config.model_type

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
                # heads_not_grouped = [
                #     i * num_heads_per_group + j for i in heads_grouped for j in range(num_heads_per_group)
                # ]
                index_to_keep_grouped = torch.LongTensor(
                    list(
                        set(range(num_grouped_heads * head_dim))
                        - {i * head_dim + j for i in heads_grouped for j in range(head_dim)}
                    )
                )
                index_to_keep_full = torch.LongTensor(
                    [i * num_heads_per_group + j for i in index_to_keep_grouped for j in range(num_heads_per_group)]
                )

                # Prune values
                attention_layer.q_proj = prune_linear_layer(attention_layer.q_proj, index_to_keep_full)
                attention_layer.k_proj = prune_linear_layer(attention_layer.k_proj, index_to_keep_grouped)
                attention_layer.v_proj = prune_linear_layer(attention_layer.v_proj, index_to_keep_grouped)
                attention_layer.o_proj = prune_linear_layer(attention_layer.o_proj, index_to_keep_full, dim=1)

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
    model, architecture = model.base_model, model.config.model_type

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
    model, architecture = model.base_model, model.config.model_type

    for layer_index, neurons in neurons_to_prune.items():
        neurons_indexes_to_keep = torch.LongTensor(list(set(range(model.config.intermediate_size)) - set(neurons)))

        if architecture == "bert":
            layer = model.encoder.layer[layer_index]

            layer.intermediate.dense = prune_linear_layer(
                layer.intermediate.dense,
                neurons_indexes_to_keep,
                dim=0,
            )
            layer.output.dense = prune_linear_layer(
                layer.output.dense,
                neurons_indexes_to_keep,
                dim=1,
            )
        elif architecture == "llama":
            layer = model.layers[layer_index].mlp

            layer.gate_proj = prune_linear_layer(
                layer.gate_proj,
                neurons_indexes_to_keep,
                dim=0,
            )
            layer.up_proj = prune_linear_layer(
                layer.up_proj,
                neurons_indexes_to_keep,
                dim=0,
            )
            layer.down_proj = prune_linear_layer(
                layer.down_proj,
                neurons_indexes_to_keep,
                dim=1,
            )
            layer.intermediate_size = len(neurons_indexes_to_keep)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # for module, dim in what_to_prune:
        #     param = layer.__getattr__(name)
        #
        #     param.dense = prune_linear_layer(
        #         param.dense,
        #         neurons_indexes_to_keep,
        #         dim=dim,
        #     )
        # # code in `prune_linear_layer` slow down the inference
        # # so re-create nn.Linear with the same weight and bias
        # weights = param.dense.weight
        # bias = param.dense.bias
        # param.dense = nn.Linear(
        #     param.dense.in_features,
        #     param.dense.out_features,
        #     device=param.dense.weight.device,
        # )
        # param.dense.weight = nn.Parameter(weights, requires_grad=weights.requires_grad)
        # param.dense.bias = nn.Parameter(bias, requires_grad=bias.requires_grad)


def prune_ffn_layers(model: PreTrainedModel, layers_to_prune: list[int]) -> None:
    """
    Prune the specified feed forward layers in the model.
    Simply replace the feed forward layers with a no-op n-0 and 0-m linear layers.

    :param model: The transformers pytorch model to prune
    :param layers_to_prune: A list of layer indices to prune
    """
    model, architecture = model.base_model, model.config.model_type

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


def select_to_prune_attention_heads(
    importance_scores: torch.Tensor,
    percent_heads_to_prune: float,
    uniform_among_layers: bool = False,
    key_value_group_size: int = 1,
    round_to_heads: int = 1,
    keep_at_least_one_head: bool = True,
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

    heads_to_prune = {}
    num_layers, num_heads = importance_scores.size()

    if uniform_among_layers:
        num_heads_to_prune = int(num_heads * percent_heads_to_prune)
        num_heads_to_prune = round(num_heads_to_prune / round_to_heads) * round_to_heads

        for layer_index in range(num_layers):
            # sort heads by importance
            importance_scores_layer = importance_scores[layer_index]
            _, sorted_indices = importance_scores_layer.sort()
            heads_to_prune_layer = sorted_indices[:num_heads_to_prune].tolist()
            heads_to_prune[layer_index] = heads_to_prune_layer

    else:
        num_heads_to_prune = int(num_layers * num_heads * percent_heads_to_prune)

        # sort heads by importance
        importance_scores_flatten = importance_scores.view(-1)
        _, sorted_indices = importance_scores_flatten.sort()
        heads_to_prune_flatten = sorted_indices[:num_heads_to_prune]

        # convert to layer-head indices
        for head_index in heads_to_prune_flatten:
            head_index = int(head_index.item())
            layer_index = head_index // num_heads
            head_index = head_index % num_heads
            heads_to_prune.setdefault(layer_index, []).append(head_index)

        # round to the nearest round_to_heads for each layer (drop excess heads)
        for layer, heads in heads_to_prune.items():
            num_heads_to_prune_layer = len(heads)
            num_heads_to_prune_layer = round(num_heads_to_prune_layer / round_to_heads) * round_to_heads
            heads_to_prune[layer] = heads[:num_heads_to_prune_layer]

    # keep at least 1 head per layer (or round_to_heads), remove unused heads_to_prune lists
    if keep_at_least_one_head:
        for layer in list(heads_to_prune.keys()):
            if len(heads_to_prune[layer]) == num_heads:
                heads_to_prune[layer] = heads_to_prune[layer][: num_heads - round_to_heads]
            if len(heads_to_prune[layer]) == 0:
                del heads_to_prune[layer]

    # expand the grouped heads to the original size
    if key_value_group_size != 1:
        heads_to_prune_expanded = {}
        for layer, heads in heads_to_prune.items():
            heads_expanded = [i * key_value_group_size + j for i in heads for j in range(key_value_group_size)]
            heads_to_prune_expanded[layer] = heads_expanded
        heads_to_prune = heads_to_prune_expanded

    return heads_to_prune


def select_to_prune_attention_layers(
    importance_scores: torch.Tensor,
    percent_layers_to_prune: float,
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
    layers_to_prune = sorted_indices[:num_layers_to_prune].tolist()

    return layers_to_prune


def select_to_prune_ffn_neurons(
    importance_scores: torch.Tensor,
    percent_neurons_to_prune: float,
    uniform_among_layers: bool = False,
    round_to: int = 1,
) -> dict[int, list[int]]:
    """
    Select least-k feed forward neurons based on the importance scores.

    :param importance_scores: The importance scores of the feed forward neurons [num_hidden_layers, intermediate_size]
    :param percent_neurons_to_prune: The percentage of feed forward neurons to keep
    :param uniform_among_layers: If True, prune the same number of neurons from each layer
    :param round_to: The number of neurons to group together for pruning (round to the nearest round_to) for gpu opts
    :return: A dictionary with the layer indices as keys and a list of neuron indices to prune as values
    """
    assert 0 <= percent_neurons_to_prune <= 1, "percent_neurons_to_prune should be in [0, 1]"

    neurons_to_prune = {}
    num_layers, num_neurons = importance_scores.size()

    if uniform_among_layers:
        num_neurons_to_prune = int(num_neurons * percent_neurons_to_prune)
        num_neurons_to_prune = round(num_neurons_to_prune / round_to) * round_to

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
            num_neurons_to_prune_layer = round(num_neurons_to_prune_layer / round_to) * round_to
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

    _, sorted_indices = importance_scores.sort()
    neurons_to_prune = sorted_indices[:num_neurons_to_prune].tolist()

    return neurons_to_prune
