import copy

import pytest
import torch
from torch import nn
from transformers import BertForMaskedLM, PreTrainedModel
from transformers.pytorch_utils import prune_linear_layer

from adaptive_pruning.nullify import (
    _nullify_embedding_layer_hidden_states,
    _nullify_layer_norm,
    _nullify_linear_layer,
    nullify_attention_heads,
    nullify_attention_layers,
    nullify_ffn_layers,
    nullify_ffn_neurons,
    nullify_hidden_state,
)
from adaptive_pruning.utils import count_zero_parameters


class TestNullifyUtils:
    @pytest.mark.parametrize("nullify_input_dim", [True, False])
    def test_nullify_linear_layer_dummy(self, nullify_input_dim: bool) -> None:
        # create a dummy linear layer
        layer = nn.Linear(10, 2, bias=False)
        layer.weight.data.fill_(1.0)

        # set params
        indexes = torch.tensor([0], dtype=torch.long)
        dummy_input = torch.ones(layer.in_features)

        # test original output
        original_output = layer(dummy_input)
        assert torch.allclose(original_output, torch.tensor([10.0, 10.0])), "Output should be 10.0 for all neurons"

        # nullify the layer
        _nullify_linear_layer(layer, indexes, input_dim=nullify_input_dim)
        nullified_output = layer(dummy_input)

        # check if the output is correct
        if nullify_input_dim:
            assert torch.allclose(nullified_output, torch.tensor([9.0, 9.0])), "Should be 9.0 as 1 input is nullified"
        else:
            assert torch.allclose(nullified_output, torch.tensor([0.0, 10.0])), "Should be 0.0 for nullified index"

    @pytest.mark.parametrize(
        "layer, indexes, nullify_input_dim",
        [
            (nn.Linear(10, 10), torch.tensor([0, 5, 2]), False),
            (nn.Linear(10, 10), torch.tensor([0, 5, 2]), True),
            (nn.Linear(10, 2), torch.tensor([1]), False),
            (nn.Linear(2, 10), torch.tensor([1]), True),
            (nn.Linear(10, 10), torch.tensor([], dtype=torch.long), False),
        ],
    )
    def test_nullify_linear_layer_output_differs(
        self, layer, indexes: torch.LongTensor, nullify_input_dim: bool
    ) -> None:
        torch.nn.init.uniform_(layer.weight, -1, 1)

        dummy_input = torch.randn(layer.in_features)

        num_zeros_in_module = count_zero_parameters(layer)
        original_output = layer(dummy_input)

        _nullify_linear_layer(layer, indexes, nullify_input_dim)
        nullified_output = layer(dummy_input)

        if indexes.shape[0] > 0:
            assert count_zero_parameters(layer) > num_zeros_in_module, "Number of zero parameters should increase"
            assert not torch.allclose(nullified_output, original_output), "Output should not be the same"
        if not nullify_input_dim:
            assert torch.allclose(
                nullified_output[nullified_output != 0], original_output[nullified_output != 0]
            ), "Output should be the same for non-nullified indices"

    @pytest.mark.parametrize(
        "layer, indexes",
        [
            (nn.Linear(10, 10), torch.tensor([0, 5, 2])),
            (nn.Linear(10, 2), torch.tensor([1])),
        ],
    )
    def test_nullify_linear_layer_same_as_prune_linear_layer(self, layer: nn.Linear, indexes: torch.LongTensor) -> None:
        torch.nn.init.uniform_(layer.weight, -1, 1)

        # make dummy input of the same shape as the layer in_features
        dummy_input = torch.randn(layer.in_features)

        # prune the layer
        pruned_layer = copy.deepcopy(layer)
        indexes_to_keep = torch.tensor([i for i in range(layer.out_features) if i not in indexes])
        pruned_layer = prune_linear_layer(pruned_layer, indexes_to_keep, dim=0)
        pruned_layer_output = pruned_layer(dummy_input)

        # nullify the layer
        _nullify_linear_layer(layer, indexes, input_dim=False)
        layer_output = layer(dummy_input)

        # check if the outputs are the same
        assert torch.allclose(pruned_layer_output, layer_output[layer_output != 0])

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

        num_zeros_in_module = count_zero_parameters(layer)
        original_output = layer(dummy_input)

        _nullify_embedding_layer_hidden_states(layer, indexes)
        nullified_output = layer(dummy_input)

        assert count_zero_parameters(layer) > num_zeros_in_module, "Number of zero parameters should increase"
        assert not torch.allclose(nullified_output, original_output), "Output should not be the same"
        assert torch.allclose(
            nullified_output[nullified_output != 0], original_output[nullified_output != 0]
        ), "Output should be the same for non-nullified indices"

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

        num_zeros_in_module = count_zero_parameters(layer)
        original_output = layer(dummy_input)

        _nullify_layer_norm(layer, indexes)
        nullified_output = layer(dummy_input)

        assert count_zero_parameters(layer) > num_zeros_in_module, "Number of zero parameters should increase"
        assert not torch.allclose(nullified_output, original_output), "Output should not be the same"


class TestNullifyAttentionHeads:
    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        heads_to_nullify = {0: [1], 1: [1]}
        head_size = test_lm_model.config.hidden_size // test_lm_model.config.num_attention_heads

        # get original output
        num_zeros_in_model = count_zero_parameters(test_lm_model)
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # nullify heads
        nullify_attention_heads(test_lm_model, heads_to_nullify)
        nullified_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # check if the outputs are different
        assert count_zero_parameters(test_lm_model) > num_zeros_in_model, "Number of zero parameters should increase"
        assert not torch.allclose(
            nullified_last_hidden_state, original_last_hidden_state
        ), "Output should not be the same"


class TestNullifyAttentionLayers:
    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        layers_to_nullify = [0, 1]

        # get original output
        num_zeros_in_model = count_zero_parameters(test_lm_model)
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # nullify layers
        nullify_attention_layers(test_lm_model, layers_to_nullify)
        nullified_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # check if the outputs are different
        assert count_zero_parameters(test_lm_model) > num_zeros_in_model, "Number of zero parameters should increase"
        assert not torch.allclose(
            nullified_last_hidden_state, original_last_hidden_state
        ), "Output should not be the same"


class TestNullifyFfnNeurons:
    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        neurons_to_nullify = {0: [1, 10, 11], 1: [0, 22, 1]}

        # get original output
        num_zeros_in_model = count_zero_parameters(test_lm_model)
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # nullify neurons
        nullify_ffn_neurons(test_lm_model, neurons_to_nullify)
        nullified_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # check if the outputs are different
        assert count_zero_parameters(test_lm_model) > num_zeros_in_model, "Number of zero parameters should increase"
        assert not torch.allclose(
            nullified_last_hidden_state, original_last_hidden_state
        ), "Output should not be the same"


class TestNullifyFfnLayers:
    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        # set params
        layers_to_nullify = [0, 1]

        # get original output
        num_zeros_in_model = count_zero_parameters(test_lm_model)
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # nullify layers
        nullify_ffn_layers(test_lm_model, layers_to_nullify)
        nullified_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # check if the outputs are different
        assert count_zero_parameters(test_lm_model) > num_zeros_in_model, "Number of zero parameters should increase"
        assert not torch.allclose(
            nullified_last_hidden_state, original_last_hidden_state
        ), "Output should not be the same"


class TestNullifyHiddenState:
    def test_differs_from_original(
        self, test_lm_model: PreTrainedModel, random_input_batch: dict[str, torch.Tensor]
    ) -> None:
        if isinstance(test_lm_model, BertForMaskedLM):
            pytest.xfail("BertForMaskedLM is not supported yet")

        # set params
        hidden_states_to_nullify = [0, 1, 31, 17, *range(50, 64)]

        # get original output
        num_zeros_in_model = count_zero_parameters(test_lm_model)
        original_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # nullify hidden states
        nullify_hidden_state(test_lm_model, hidden_states_to_nullify)
        nullified_last_hidden_state = test_lm_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]

        # check if the outputs are different
        assert count_zero_parameters(test_lm_model) > num_zeros_in_model, "Number of zero parameters should increase"
        assert not torch.allclose(
            nullified_last_hidden_state, original_last_hidden_state
        ), "Output should not be the same"
