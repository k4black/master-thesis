import copy

import pytest
from transformers import PreTrainedModel
import torch

from adaptive_pruning.pruning import (
    prune_attention_heads, prune_attention_layers, prune_ffn_neurons, prune_ffn_layers, prune_hidden_state
)
from adaptive_pruning.utils import (
    nullify_attention_heads, nullify_attention_layers, nullify_ffn_neurons, nullify_ffn_layers, nullify_hidden_state
)


class TestPruneAttentionHeads:
    @staticmethod
    def _assert_attention_heads_number_by_layer(model: PreTrainedModel, expected_heads: list[int]) -> None:
        head_size = model.config.hidden_size // model.config.num_attention_heads

        for i, layer in enumerate(model.encoder.layer):
            assert layer.attention.self.num_attention_heads == expected_heads[i], \
                f"Layer {i}: expected {expected_heads[i]} heads, got {layer.attention.self.num_attention_heads}"

            assert layer.attention.self.query.out_features == head_size * expected_heads[i], f"Layer {i}"
            assert layer.attention.output.dense.in_features == head_size * expected_heads[i], f"Layer {i}"

    @pytest.mark.parametrize(
        "heads_to_prune, expected_heads",
        [
            ({}, [2, 2]),
            ({0: [0]}, [1, 2]),
            ({0: [0], 1: [0]}, [1, 1]),
            ({0: [0, 1], 1: [0, 1]}, [0, 0]),
        ],
    )
    def test_num_heads(
        self, bert_tiny_model: PreTrainedModel, heads_to_prune: dict[int, list[int]], expected_heads: list[int],
    ) -> None:
        self._assert_attention_heads_number_by_layer(bert_tiny_model, [2, 2])

        prune_attention_heads(bert_tiny_model, heads_to_prune)

        self._assert_attention_heads_number_by_layer(bert_tiny_model, expected_heads)

    def test_same_as_nullify(self, bert_tiny_model: PreTrainedModel, random_input: dict[str, torch.Tensor]) -> None:
        
        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # set params
            head_index = 0
            heads_to_prune = {0: [head_index]}

            # nullify first head of the first layer
            nullified_model = copy.deepcopy(bert_tiny_model)
            nullify_attention_heads(nullified_model, heads_to_prune)
            # get output of the nullified model
            nullified_last_hidden_state = nullified_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # check that the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # prune the first head of the first layer
            prune_attention_heads(bert_tiny_model, heads_to_prune)
            # get output of the pruned model
            pruned_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # check that the output of the pruned model is different from the original model
            assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

            # check that the output of the pruned model is the same as the nullified model
            assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state)


class TestAttentionLayerPruning:
    @staticmethod
    def _assert_attention_layers_existence(model: PreTrainedModel, expected_layers: list[bool]) -> None:
        for i, layer in enumerate(model.encoder.layer):
            if expected_layers[i]:
                assert layer.attention.output.dense.in_features == model.config.hidden_size
            else:
                assert layer.attention.output.dense.in_features == 0

    @pytest.mark.parametrize(
        "layers_to_prune, expected_layers",
        [
            ([], [True, True]),
            ([0], [False, True]),
            ([0, 1], [False, False]),
        ],
    )
    def test_layer_pruning(
        self, bert_tiny_model: PreTrainedModel, layers_to_prune: list[int], expected_layers: list[bool],
    ) -> None:
        self._assert_attention_layers_existence(bert_tiny_model, [True, True])

        prune_attention_layers(bert_tiny_model, layers_to_prune)

        self._assert_attention_layers_existence(bert_tiny_model, expected_layers)

    def test_same_as_nullify(self, bert_tiny_model: PreTrainedModel, random_input: dict[str, torch.Tensor]) -> None:
        
        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # set params
            layer_index = 0
            layers_to_prune = [layer_index]

            # nullify first layer
            nullified_model = copy.deepcopy(bert_tiny_model)
            nullify_attention_layers(nullified_model, layers_to_prune)
            # get output of the nullified model
            nullified_last_hidden_state = nullified_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # check that the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # prune the first layer
            prune_attention_layers(bert_tiny_model, layers_to_prune)
            # get output of the pruned model
            pruned_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # check that the output of the pruned model is different from the original model
            assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

            # check that the output of the pruned model is the same as the nullified model
            assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state)


class TestPruneFeedForwardNeurons:
    @staticmethod
    def _assert_feed_forward_neurons_number_by_layer(model: PreTrainedModel, expected_neurons: list[int]) -> None:
        for i, layer in enumerate(model.encoder.layer):
            assert layer.intermediate.dense.out_features == expected_neurons[i], \
                f"Layer {i} expected {expected_neurons[i]} neurons, got {layer.intermediate.dense.out_features}"
            assert layer.output.dense.in_features == expected_neurons[i], \
                f"Layer {i} expected {expected_neurons[i]} neurons, got {layer.output.dense.in_features}"

    @pytest.mark.parametrize(
        "neurons_to_prune, expected_neurons",
        [
            ({}, [512, 512]),
            ({0: [0]}, [511, 512]),
            ({0: [0, 211]}, [510, 512]),
            ({0: [50, 21], 1: [20, 21]}, [510, 510]),
            ({0: [0, 8, 10, 31], 1: [0, 3, 5, 7, 510, 511]}, [508, 506]),
        ],
    )
    def test_num_neurons(
        self, bert_tiny_model: PreTrainedModel, neurons_to_prune: dict[int, list[int]], expected_neurons: list[int],
    ) -> None:
        self._assert_feed_forward_neurons_number_by_layer(bert_tiny_model, [512, 512])

        prune_ffn_neurons(bert_tiny_model, neurons_to_prune)

        self._assert_feed_forward_neurons_number_by_layer(bert_tiny_model, expected_neurons)

    def test_same_as_nullify(self, bert_tiny_model: PreTrainedModel, random_input: dict[str, torch.Tensor]) -> None:
        
        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # set params
            neurons_to_prune = {0: [0, 10, *range(50, 60), 511]}

            # nullify first neuron of the first layer
            nullified_model = copy.deepcopy(bert_tiny_model)
            nullify_ffn_neurons(nullified_model, neurons_to_prune)
            # get output of the nullified model
            nullified_last_hidden_state = nullified_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # check that the output of the nullified model is different from the original model
            assert original_last_hidden_state.shape == nullified_last_hidden_state.shape
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state, atol=1e-5)

            # prune the first neuron of the first layer
            prune_ffn_neurons(bert_tiny_model, neurons_to_prune)
            # get output of the pruned model
            pruned_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # check that the output of the pruned model is different from the original model
            assert original_last_hidden_state.shape == pruned_last_hidden_state.shape
            assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state, atol=1e-5)

            # check that the output of the pruned model is the same as the nullified model
            assert nullified_last_hidden_state.shape == pruned_last_hidden_state.shape
            assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state, atol=1e-5)


class TestPruneFeedForwardLayers:
    @staticmethod
    def _assert_feed_forward_layers_existence(model: PreTrainedModel, expected_layers: list[bool]) -> None:
        for i, layer in enumerate(model.encoder.layer):
            if expected_layers[i]:
                assert layer.intermediate.dense.out_features == model.config.intermediate_size, \
                    (f"Layer {i} expected {model.config.intermediate_size} neurons, "
                     f"got {layer.intermediate.dense.out_features}")
                assert layer.output.dense.in_features == model.config.intermediate_size, \
                    f"Layer {i} expected {model.config.intermediate_size} neurons, got {layer.output.dense.in_features}"
            else:
                assert layer.intermediate.dense.out_features == 0, \
                    f"Layer {i} expected 0 neurons, got {layer.intermediate.dense.out_features}"
                assert layer.output.dense.in_features == 0, \
                    f"Layer {i} expected 0 neurons, got {layer.output.dense.in_features}"

    @pytest.mark.parametrize(
        "layers_to_prune, expected_layers",
        [
            ([], [True, True]),
            ([0], [False, True]),
            ([0, 1], [False, False]),
        ],
    )
    def test_layer_pruning(
        self, bert_tiny_model: PreTrainedModel, layers_to_prune: list[int], expected_layers: list[bool],
    ) -> None:
        self._assert_feed_forward_layers_existence(bert_tiny_model, [True, True])

        prune_ffn_layers(bert_tiny_model, layers_to_prune)

        self._assert_feed_forward_layers_existence(bert_tiny_model, expected_layers)

    def test_same_as_nullify(self, bert_tiny_model: PreTrainedModel, random_input: dict[str, torch.Tensor]) -> None:
        
        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # set params
            layers_to_prune = [0]

            # nullify first layer
            nullified_model = copy.deepcopy(bert_tiny_model)
            nullify_ffn_layers(nullified_model, layers_to_prune)
            # get output of the nullified model
            nullified_last_hidden_state = nullified_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # check that the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # prune the first layer
            prune_ffn_layers(bert_tiny_model, layers_to_prune)
            # get output of the pruned model
            pruned_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]

            # check that the output of the pruned model is different from the original model
            assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

            # check that the output of the pruned model is the same as the nullified model
            assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state)


class TestPruneHiddenState:
    @staticmethod
    def _assert_hidden_state_neurons_number(model: PreTrainedModel, expected_hidden_state: int) -> None:
        assert model.config.hidden_size == expected_hidden_state

        # embedding layer
        assert model.embeddings.word_embeddings.weight.shape[1] == expected_hidden_state
        assert model.embeddings.position_embeddings.weight.shape[1] == expected_hidden_state
        assert model.embeddings.token_type_embeddings.weight.shape[1] == expected_hidden_state

        # main layers
        for layer in model.encoder.layer:
            assert layer.attention.self.query.in_features == expected_hidden_state
            assert layer.attention.self.key.in_features == expected_hidden_state
            assert layer.attention.self.value.in_features == expected_hidden_state
            assert layer.intermediate.dense.in_features == expected_hidden_state
            assert layer.output.dense.out_features == expected_hidden_state

        # pooler
        assert model.pooler.dense.in_features == expected_hidden_state

    @pytest.mark.parametrize(
        "neurons_to_prune, expected_hidden_state",
        [
            ([], 128),
            ([0], 127),
            ([0, 1], 126),
            ([0, 1, 2], 125),
            (list(range(120)), 8),
            ([0, 1, 127, *range(10, 20)], 115),
        ]
    )
    def test_num_neurons(
        self, bert_tiny_model: PreTrainedModel, neurons_to_prune: list[int], expected_hidden_state: int,
    ) -> None:
        self._assert_hidden_state_neurons_number(bert_tiny_model, 128)

        prune_hidden_state(bert_tiny_model, neurons_to_prune)

        self._assert_hidden_state_neurons_number(bert_tiny_model, expected_hidden_state)

    def test_pass_with_correct_dim(self, bert_tiny_model: PreTrainedModel, random_input: dict[str, torch.Tensor]) -> None:
        # get output of the original model
        original_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]
        assert original_last_hidden_state.shape == (*random_input["input_ids"].shape, bert_tiny_model.config.hidden_size)

        # set params
        neurons_to_prune = [0, 1, 127, *range(10, 20)]

        # prune the hidden state
        prune_hidden_state(bert_tiny_model, neurons_to_prune)
        # get output of the pruned model
        pruned_last_hidden_state = bert_tiny_model(random_input["input_ids"], random_input["attention_mask"])[0]
        assert pruned_last_hidden_state.shape == (*random_input["input_ids"].shape, 115)
