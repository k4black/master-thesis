import copy

import pytest
import torch
from torch import nn
from transformers import BertForMaskedLM, PreTrainedModel

from adaptive_pruning.nullify import (
    nullify_attention_heads,
    nullify_attention_layers,
    nullify_ffn_layers,
    nullify_ffn_neurons,
)
from adaptive_pruning.pruning import (
    _prune_embedding_layer_hidden_states,
    _prune_layer_norm,
    _prune_linear_layer,
    prune_attention_heads,
    prune_attention_layers,
    prune_ffn_layers,
    prune_ffn_neurons,
    prune_hidden_states,
)
from adaptive_pruning.utils import count_total_parameters


class TestPruneUtils:
    @pytest.mark.parametrize(
        "prune_input_dim, pruned_input, expected_output",
        [
            (True, torch.ones(9), torch.tensor([9.0, 9.0])),
            (False, torch.ones(10), torch.tensor([10.0])),
        ],
    )
    def test_prune_linear_layer_dummy(
        self, prune_input_dim: bool, pruned_input: torch.Tensor, expected_output: torch.Tensor
    ) -> None:
        # create a dummy linear layer
        layer = nn.Linear(10, 2, bias=False)
        layer.weight.data.fill_(1.0)

        # set params
        indexes = torch.tensor([0], dtype=torch.long)

        # nullify the layer
        layer = _prune_linear_layer(layer, indexes, input_dim=prune_input_dim)
        pruned_output = layer(pruned_input)

        # check if the output is correct
        assert torch.allclose(pruned_output, expected_output)

    @pytest.mark.parametrize(
        "layer, indexes, prune_input_dim",
        [
            (nn.Linear(10, 10), torch.tensor([0, 5, 2]), False),
            (nn.Linear(10, 10), torch.tensor([0, 5, 2]), True),
            (nn.Linear(10, 2), torch.tensor([1]), False),
            (nn.Linear(2, 10), torch.tensor([1]), True),
            (nn.Linear(10, 10), torch.tensor([], dtype=torch.long), False),
        ],
    )
    def test_nullify_linear_layer_output_differs(self, layer, indexes: torch.LongTensor, prune_input_dim) -> None:
        torch.nn.init.uniform_(layer.weight, -1, 1)

        dummy_input = torch.randn(layer.in_features)
        original_output = layer(dummy_input)

        if prune_input_dim:
            pruned_input = torch.randn(layer.in_features - indexes.shape[0])
        else:
            pruned_input = torch.randn(layer.in_features)

        layer = _prune_linear_layer(layer, indexes, prune_input_dim)
        pruned_output = layer(pruned_input)

        if indexes.shape[0] > 0:
            if prune_input_dim:
                assert pruned_output.shape == original_output.shape, "Output shape should NOT change"
            else:
                assert pruned_output.shape != original_output.shape, "Output shape should change"

    @pytest.mark.parametrize(
        "layer, indexes",
        [
            (nn.Embedding(100, 32), torch.tensor([0, 5, 2])),
            (nn.Embedding(10, 16), torch.tensor([15])),
            (nn.Embedding(10, 8), torch.tensor([0, 5, 2])),
            (nn.Embedding(1000, 32), torch.tensor([0, 31])),
        ],
    )
    def test_nullify_embedding_layer_output_differs(self, layer: nn.Embedding, indexes: torch.LongTensor) -> None:
        torch.nn.init.uniform_(layer.weight, -1, 1)

        dummy_input = torch.tensor([1, 7, 3, 9, 9, 0])

        original_output = layer(dummy_input)

        layer = _prune_embedding_layer_hidden_states(layer, indexes)
        nullified_output = layer(dummy_input)

        assert nullified_output.shape[0] == original_output.shape[0], "Number of tokens should not change"
        assert nullified_output.shape[1] != original_output.shape[1], "Output shape of hidden size should change"

    @pytest.mark.parametrize(
        "layer, indexes",
        [
            (nn.LayerNorm(100), torch.tensor([0, 5, 2, 98])),
            (nn.LayerNorm(10), torch.tensor([9])),
            (nn.LayerNorm(10), torch.tensor([0, 5, 2])),
        ],
    )
    def test_nullify_layer_norm_output_differs(self, layer: nn.LayerNorm, indexes: torch.LongTensor) -> None:
        torch.nn.init.uniform_(layer.weight, -1, 1)

        dummy_input = torch.randn(layer.normalized_shape)

        original_output = layer(dummy_input)

        to_keep_indexes = torch.tensor([i for i in range(layer.normalized_shape[0]) if i not in indexes])
        pruned_input = dummy_input[to_keep_indexes]

        layer = _prune_layer_norm(layer, indexes)
        pruned_output = layer(pruned_input)

        assert original_output.shape != pruned_output.shape, "Output shape should change"


class TestPruneAttentionHeads:
    @staticmethod
    def _assert_attention_heads_number_by_layer(model: PreTrainedModel, expected_heads: list[int]) -> None:
        model = model.base_model
        head_size = model.config.hidden_size // model.config.num_attention_heads

        if model.config.model_type == "bert":
            for i, layer in enumerate(model.encoder.layer):
                assert (
                    layer.attention.self.num_attention_heads == expected_heads[i]
                ), f"Layer {i}: expected {expected_heads[i]} heads, got {layer.attention.self.num_attention_heads}"

                assert layer.attention.self.query.out_features == head_size * expected_heads[i], f"Layer {i}"
                assert layer.attention.output.dense.in_features == head_size * expected_heads[i], f"Layer {i}"

        elif model.config.model_type == "llama":
            num_heads_per_group = model.config.num_attention_heads // model.config.num_key_value_heads
            for i, layer in enumerate(model.layers):
                assert (
                    layer.self_attn.num_heads == expected_heads[i]
                ), f"Layer {i}: expected {expected_heads[i]} heads, got {layer.self_attn.num_heads}"
                assert (
                    layer.self_attn.num_key_value_heads == expected_heads[i] // num_heads_per_group
                ), f"Layer {i}: expected {expected_heads[i] // num_heads_per_group} key-value heads, got {layer.self_attn.num_key_value_heads}"

                assert layer.self_attn.q_proj.out_features == head_size * expected_heads[i], f"Layer {i} q_proj error"
                assert (
                    layer.self_attn.k_proj.out_features == head_size * expected_heads[i] // num_heads_per_group
                ), f"Layer {i} k_proj error"
                assert (
                    layer.self_attn.v_proj.out_features == head_size * expected_heads[i] // num_heads_per_group
                ), f"Layer {i} v_proj error"
                assert layer.self_attn.o_proj.in_features == head_size * expected_heads[i], f"Layer {i} o_proj error"

    @pytest.mark.parametrize(
        "heads_to_prune, expected_heads",
        [
            ({}, [4, 4]),
            ({0: [0]}, [3, 4]),
            ({0: [3]}, [3, 4]),
            ({0: [0], 1: [0]}, [3, 3]),
            ({0: [0, 1], 1: [0, 1]}, [2, 2]),
            ({0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}, [0, 0]),
        ],
    )
    def test_num_heads_bert(
        self,
        test_lm_model_bert: PreTrainedModel,
        heads_to_prune: dict[int, list[int]],
        expected_heads: list[int],
    ) -> None:
        self._assert_attention_heads_number_by_layer(test_lm_model_bert, [4, 4])

        prune_attention_heads(test_lm_model_bert, heads_to_prune)

        self._assert_attention_heads_number_by_layer(test_lm_model_bert, expected_heads)

    # grouped heads are used, so round to prune grouped heads only as the structure
    @pytest.mark.parametrize(
        "heads_to_prune, expected_heads",
        [
            ({}, [4, 4]),
            ({0: [0]}, [2, 4]),
            ({0: [3]}, [2, 4]),
            ({0: [0], 1: [0]}, [2, 2]),
            ({0: [0, 1], 1: [0, 3]}, [2, 0]),
            ({0: [0, 1, 2, 3], 1: [0, 1, 2, 3]}, [0, 0]),
        ],
    )
    def test_num_heads_llama(
        self,
        test_lm_model_llama: PreTrainedModel,
        heads_to_prune: dict[int, list[int]],
        expected_heads: list[int],
    ) -> None:
        self._assert_attention_heads_number_by_layer(test_lm_model_llama, [4, 4])

        prune_attention_heads(test_lm_model_llama, heads_to_prune)

        self._assert_attention_heads_number_by_layer(test_lm_model_llama, expected_heads)

    @pytest.mark.parametrize(
        "heads_to_prune",
        [
            {0: [0]},
            {0: [0, 1, 2, 3]},
            {0: [1], 1: [0, 3]},
            {0: [0, 1, 2, 3], 1: [0, 1, 2, 3]},
        ],
    )
    def test_less_params(self, test_lm_model: PreTrainedModel, heads_to_prune: dict[int, list[int]]) -> None:
        params_before = count_total_parameters(test_lm_model)

        prune_attention_heads(test_lm_model, heads_to_prune)

        assert count_total_parameters(test_lm_model) < params_before

    def test_requires_grad_bert(self, test_base_model_bert: PreTrainedModel) -> None:
        # set params
        heads_to_prune = {0: [0]}

        # save old requires_grad
        old_requires_grad_query = test_base_model_bert.encoder.layer[0].attention.self.query.weight.requires_grad
        old_requires_grad_bias = test_base_model_bert.encoder.layer[0].attention.self.query.bias.requires_grad

        # prune
        prune_attention_heads(test_base_model_bert, heads_to_prune)

        # check that the model requires grad
        assert (
            test_base_model_bert.encoder.layer[0].attention.self.query.weight.requires_grad == old_requires_grad_query
        )
        assert test_base_model_bert.encoder.layer[0].attention.self.query.bias.requires_grad == old_requires_grad_bias

    @torch.no_grad()
    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        heads_to_prune = {0: [1]}

        # get output of the original model
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # prune
        prune_attention_heads(test_lm_model, heads_to_prune)
        pruned_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[
            0
        ]

        # check the output of the pruned model is different from the original model
        assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

    @torch.no_grad()
    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]) -> None:
        if isinstance(test_lm_model, BertForMaskedLM):
            pytest.xfail("BertForMaskedLM does not support nullify_attention_heads for now")

        # set params
        heads_to_prune = {0: [1]}

        # nullify first head of the first layer
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_attention_heads(nullified_model, heads_to_prune)
        # get output of the nullified model
        nullified_last_hidden_state = nullified_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # prune the first head of the first layer
        prune_attention_heads(test_lm_model, heads_to_prune)
        # get output of the pruned model
        pruned_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[
            0
        ]

        # check the output of the pruned model is the same as the nullified model
        assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state)


class TestAttentionLayerPruning:
    @staticmethod
    def _assert_attention_layers_existence(model: PreTrainedModel, expected_layers: list[bool]) -> None:
        model, architecture = model.base_model, model.config.model_type

        if architecture == "bert":
            for i, layer in enumerate(model.encoder.layer):
                if expected_layers[i]:
                    assert layer.attention.output.dense.in_features == model.config.hidden_size
                else:
                    assert layer.attention.output.dense.in_features == 0
        elif architecture == "llama":
            for i, layer in enumerate(model.layers):
                if expected_layers[i]:
                    assert layer.self_attn.o_proj.in_features == model.config.hidden_size
                else:
                    assert layer.self_attn.o_proj.in_features == 0
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    @pytest.mark.parametrize(
        "layers_to_prune, expected_layers",
        [
            ([], [True, True]),
            ([0], [False, True]),
            ([0, 1], [False, False]),
        ],
    )
    def test_layer_pruning(
        self,
        test_lm_model: PreTrainedModel,
        layers_to_prune: list[int],
        expected_layers: list[bool],
    ) -> None:
        self._assert_attention_layers_existence(test_lm_model, [True, True])

        prune_attention_layers(test_lm_model, layers_to_prune)

        self._assert_attention_layers_existence(test_lm_model, expected_layers)

    def test_requires_grad_bert(self, test_base_model_bert: PreTrainedModel) -> None:
        # set params
        layers_to_prune = [0]

        # save old requires_grad
        old_requires_grad = test_base_model_bert.encoder.layer[0].attention.output.dense.weight.requires_grad

        # prune
        prune_attention_layers(test_base_model_bert, layers_to_prune)

        # check that the model requires grad
        assert test_base_model_bert.encoder.layer[0].attention.output.dense.weight.requires_grad == old_requires_grad

    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        layers_to_prune = [0]

        # get output of the original model
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # prune
        prune_attention_layers(test_lm_model, layers_to_prune)
        pruned_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[
            0
        ]

        # check the output of the pruned model is different from the original model
        assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

    @torch.no_grad()
    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]) -> None:
        # set params
        layers_to_prune = [1]

        # nullify first layer
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_attention_layers(nullified_model, layers_to_prune)
        # get output of the nullified model
        nullified_last_hidden_state = nullified_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # prune the first layer
        prune_attention_layers(test_lm_model, layers_to_prune)
        # get output of the pruned model
        pruned_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[
            0
        ]

        # check the output of the pruned model is the same as the nullified model
        assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state)


class TestPruneFeedForwardNeurons:
    @staticmethod
    def _assert_feed_forward_neurons_number_by_layer(model: PreTrainedModel, expected_neurons: list[int]) -> None:
        model, architecture = model.base_model, model.config.model_type

        if architecture == "bert":
            for i, layer in enumerate(model.encoder.layer):
                assert (
                    layer.intermediate.dense.out_features == expected_neurons[i]
                ), f"Layer {i} expected {expected_neurons[i]} neurons, got {layer.intermediate.dense.out_features}"
                assert (
                    layer.output.dense.in_features == expected_neurons[i]
                ), f"Layer {i} expected {expected_neurons[i]} neurons, got {layer.output.dense.in_features}"

        elif architecture == "llama":
            for i, layer in enumerate(model.layers):
                assert (
                    layer.mlp.gate_proj.out_features == expected_neurons[i]
                ), f"Layer {i} expected {expected_neurons[i]} neurons, got {layer.mlp.gate_proj.out_features}"
                assert (
                    layer.mlp.up_proj.out_features == expected_neurons[i]
                ), f"Layer {i} expected {expected_neurons[i]} neurons, got {layer.mlp.up_proj.out_features}"
                assert (
                    layer.mlp.down_proj.in_features == expected_neurons[i]
                ), f"Layer {i} expected {expected_neurons[i]} neurons, got {layer.mlp.down_proj.in_features}"

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    @pytest.mark.parametrize(
        "neurons_to_prune, expected_neurons",
        [
            ({}, [128, 128]),
            ({0: [0]}, [127, 128]),
            ({0: [0, 120]}, [126, 128]),
            ({0: [50, 21], 1: [20, 21]}, [126, 126]),
            ({0: [0, 8, 10, 31], 1: [0, 3, 5, 7, 126, 127]}, [124, 122]),
        ],
    )
    def test_num_neurons(
        self,
        test_lm_model: PreTrainedModel,
        neurons_to_prune: dict[int, list[int]],
        expected_neurons: list[int],
    ) -> None:
        self._assert_feed_forward_neurons_number_by_layer(test_lm_model, [128, 128])

        prune_ffn_neurons(test_lm_model, neurons_to_prune)

        self._assert_feed_forward_neurons_number_by_layer(test_lm_model, expected_neurons)

    def test_requires_grad_bert(self, test_base_model_bert: PreTrainedModel) -> None:
        # set params
        neurons_to_prune = {0: [0, 10, *range(50, 60), 127]}

        # save old requires_grad
        old_requires_grad_intermediate = test_base_model_bert.encoder.layer[0].intermediate.dense.weight.requires_grad
        old_requires_grad_output = test_base_model_bert.encoder.layer[0].output.dense.weight.requires_grad

        # prune
        prune_ffn_neurons(test_base_model_bert, neurons_to_prune)

        # check that the model requires grad
        assert (
            test_base_model_bert.encoder.layer[0].intermediate.dense.weight.requires_grad
            == old_requires_grad_intermediate
        )
        assert test_base_model_bert.encoder.layer[0].output.dense.weight.requires_grad == old_requires_grad_output

    @torch.no_grad()
    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        neurons_to_prune = {0: [0, 10, *range(50, 60), 127]}

        # get output of the original model
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # prune
        prune_ffn_neurons(test_lm_model, neurons_to_prune)
        pruned_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[
            0
        ]

        # check the output of the pruned model is different from the original model
        assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

    @torch.no_grad()
    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:
        # set params
        neurons_to_prune = {0: [0, 10, *range(50, 60), 127]}

        # nullify first neuron of the first layer
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_ffn_neurons(nullified_model, neurons_to_prune)
        # get output of the nullified model
        nullified_last_hidden_state = nullified_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # prune the first neuron of the first layer
        prune_ffn_neurons(test_lm_model, neurons_to_prune)
        # get output of the pruned model
        pruned_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[
            0
        ]

        # check the output of the pruned model is the same as the nullified model
        assert nullified_last_hidden_state.shape == pruned_last_hidden_state.shape
        assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state, atol=1e-5)


class TestPruneFeedForwardLayers:
    @staticmethod
    def _assert_feed_forward_layers_existence(model: PreTrainedModel, expected_layers: list[bool]) -> None:
        model, architecture = model.base_model, model.config.model_type

        if architecture == "bert":
            for i, layer in enumerate(model.encoder.layer):
                if expected_layers[i]:
                    assert layer.intermediate.dense.out_features == model.config.intermediate_size, (
                        f"Layer {i} expected {model.config.intermediate_size} neurons, "
                        f"got {layer.intermediate.dense.out_features}"
                    )
                    assert (
                        layer.output.dense.in_features == model.config.intermediate_size
                    ), f"Layer {i} expected {model.config.intermediate_size} neurons, got {layer.output.dense.in_features}"
                else:
                    assert (
                        layer.intermediate.dense.out_features == 0
                    ), f"Layer {i} expected 0 neurons, got {layer.intermediate.dense.out_features}"
                    assert (
                        layer.output.dense.in_features == 0
                    ), f"Layer {i} expected 0 neurons, got {layer.output.dense.in_features}"
        elif architecture == "llama":
            for i, layer in enumerate(model.layers):
                if expected_layers[i]:
                    assert layer.mlp.gate_proj.out_features == model.config.intermediate_size, (
                        f"Layer {i} expected {model.config.intermediate_size} neurons, "
                        f"got {layer.mlp.gate_proj.out_features}"
                    )
                    assert layer.mlp.up_proj.out_features == model.config.intermediate_size, (
                        f"Layer {i} expected {model.config.intermediate_size} neurons, "
                        f"got {layer.mlp.up_proj.out_features}"
                    )
                    assert layer.mlp.down_proj.in_features == model.config.intermediate_size, (
                        f"Layer {i} expected {model.config.intermediate_size} neurons, "
                        f"got {layer.mlp.down_proj.in_features}"
                    )
                else:
                    assert (
                        layer.mlp.gate_proj.out_features == 0
                    ), f"Layer {i} expected 0 neurons, got {layer.mlp.gate_proj.out_features}"
                    assert (
                        layer.mlp.up_proj.out_features == 0
                    ), f"Layer {i} expected 0 neurons, got {layer.mlp.up_proj.out_features}"
                    assert (
                        layer.mlp.down_proj.in_features == 0
                    ), f"Layer {i} expected 0 neurons, got {layer.mlp.down_proj.in_features}"

    @pytest.mark.parametrize(
        "layers_to_prune, expected_layers",
        [
            ([], [True, True]),
            ([0], [False, True]),
            ([0, 1], [False, False]),
        ],
    )
    def test_layer_pruning(
        self,
        test_lm_model: PreTrainedModel,
        layers_to_prune: list[int],
        expected_layers: list[bool],
    ) -> None:
        self._assert_feed_forward_layers_existence(test_lm_model, [True, True])

        prune_ffn_layers(test_lm_model, layers_to_prune)

        self._assert_feed_forward_layers_existence(test_lm_model, expected_layers)

    def test_requires_grad_bert(self, test_base_model_bert: PreTrainedModel) -> None:
        # set params
        layers_to_prune = [0]

        # save old requires_grad
        old_requires_grad_intermediate = test_base_model_bert.encoder.layer[0].intermediate.dense.weight.requires_grad
        old_requires_grad_output = test_base_model_bert.encoder.layer[0].output.dense.weight.requires_grad

        # prune
        prune_ffn_layers(test_base_model_bert, layers_to_prune)

        # check that the model requires grad
        assert (
            test_base_model_bert.encoder.layer[0].intermediate.dense.weight.requires_grad
            == old_requires_grad_intermediate
        )
        assert test_base_model_bert.encoder.layer[0].output.dense.weight.requires_grad == old_requires_grad_output

    @torch.no_grad()
    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        layers_to_prune = [1]

        # get output of the original model
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # prune
        prune_ffn_layers(test_lm_model, layers_to_prune)
        pruned_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[
            0
        ]

        # check the output of the pruned model is different from the original model
        assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]) -> None:
        # set params
        layers_to_prune = [1]

        # nullify first layer
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_ffn_layers(nullified_model, layers_to_prune)
        # get output of the nullified model
        nullified_last_hidden_state = nullified_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # prune the first layer
        prune_ffn_layers(test_lm_model, layers_to_prune)
        # get output of the pruned model
        pruned_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[
            0
        ]

        # check the output of the pruned model is the same as the nullified model
        assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state)


class TestPruneHiddenState:
    @staticmethod
    def _assert_hidden_size_by_neurons_number(model: PreTrainedModel, expected_hidden_size: int) -> None:
        model, architecture = model.base_model, model.config.model_type

        assert model.config.hidden_size == expected_hidden_size

        if architecture == "bert":
            # embedding layer
            assert model.embeddings.word_embeddings.weight.shape[1] == expected_hidden_size
            assert model.embeddings.position_embeddings.weight.shape[1] == expected_hidden_size
            assert model.embeddings.token_type_embeddings.weight.shape[1] == expected_hidden_size

            # main layers
            for layer in model.encoder.layer:
                assert layer.attention.self.query.in_features == expected_hidden_size
                assert layer.attention.self.key.in_features == expected_hidden_size
                assert layer.attention.self.value.in_features == expected_hidden_size
                assert layer.intermediate.dense.in_features == expected_hidden_size
                assert layer.output.dense.out_features == expected_hidden_size

            # pooler
            assert model.pooler.dense.in_features == expected_hidden_size
        elif architecture == "llama":
            # embedding layer
            assert model.embed_tokens.weight.shape[1] == expected_hidden_size, "embed_tokens"

            # norms
            assert model.norm.weight.shape[0] == expected_hidden_size, "norm"

            # main layers
            for layer in model.layers:
                assert layer.input_layernorm.weight.shape[0] == expected_hidden_size, "input_layernorm"
                assert (
                    layer.post_attention_layernorm.weight.shape[0] == expected_hidden_size
                ), "post_attention_layernorm"
                assert layer.self_attn.q_proj.in_features == expected_hidden_size, "q_proj"
                assert layer.self_attn.k_proj.in_features == expected_hidden_size, "k_proj"
                assert layer.self_attn.v_proj.in_features == expected_hidden_size, "v_proj"
                assert layer.self_attn.o_proj.out_features == expected_hidden_size, "o_proj"
                assert layer.mlp.gate_proj.in_features == expected_hidden_size, "gate_proj"
                assert layer.mlp.up_proj.in_features == expected_hidden_size, "up_proj"
                assert layer.mlp.down_proj.out_features == expected_hidden_size, "down_proj"
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    @pytest.mark.parametrize(
        "neurons_to_prune, expected_hidden_size",
        [
            ([], 64),
            ([0], 63),
            ([0, 1], 62),
            ([0, 1, 2], 61),
            (list(range(60)), 4),
            ([0, 1, 51, *range(10, 20)], 51),
        ],
    )
    def test_num_hidden_state_neurons(
        self,
        test_base_model: PreTrainedModel,
        neurons_to_prune: list[int],
        expected_hidden_size,
    ) -> None:
        self._assert_hidden_size_by_neurons_number(test_base_model, 64)

        prune_hidden_states(test_base_model, neurons_to_prune)

        self._assert_hidden_size_by_neurons_number(test_base_model, expected_hidden_size)

    def test_pass_with_correct_dim(
        self, test_base_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        neurons_to_prune = [0, 1, 63, *range(10, 20)]

        # get output of the original model
        original_last_hidden_state = test_base_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]
        assert original_last_hidden_state.shape == (
            *random_input_batch["input_ids"].shape,
            test_base_model.config.hidden_size,
        )

        # prune the hidden state
        prune_hidden_states(test_base_model, neurons_to_prune)
        # get output of the pruned model
        pruned_last_hidden_state = test_base_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]
        assert pruned_last_hidden_state.shape == (*random_input_batch["input_ids"].shape, 51)
