import copy

import pytest
import torch
from transformers import BertForMaskedLM, PreTrainedModel

from adaptive_pruning.injections import (
    inject_attention_head_mask,
    inject_attention_layer_mask,
    inject_ffn_layer_mask,
    inject_ffn_neuron_mask,
    inject_hidden_state_mask,
)
from adaptive_pruning.nullify import (
    nullify_attention_heads,
    nullify_attention_layers,
    nullify_ffn_layers,
    nullify_ffn_neurons,
    nullify_hidden_states,
)


class TestInjectAttentionHeadMask:

    def _get_test_mask(self, model: PreTrainedModel, heads_to_mask: dict[int, list[int]]) -> torch.Tensor:
        head_mask = torch.ones((model.config.num_hidden_layers, model.config.num_attention_heads), device=model.device)
        for layer, heads in heads_to_mask.items():
            head_mask[layer, heads] = 0
        return head_mask

    @torch.no_grad()
    def test_differs_from_original(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        # set params
        heads_to_mask = {0: [0]}
        head_mask = self._get_test_mask(test_lm_model, heads_to_mask)

        # get output of the original model
        original_last_hidden_state = test_lm_model(**random_batch)[0]

        # inject mask
        masking_handles = inject_attention_head_mask(test_lm_model, head_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is different from the original model
        assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

        # remove mask via handles
        for handle in masking_handles:
            handle.remove()
        no_handles_last_hidden_state = test_lm_model(**random_batch)[0]

        assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)

    @torch.no_grad()
    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        # set params
        heads_to_mask = {0: [0]}
        head_mask = self._get_test_mask(test_lm_model, heads_to_mask)

        # nullify parts of the model
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_attention_heads(nullified_model, heads_to_mask)
        nullified_last_hidden_state = nullified_model(**random_batch)[0]

        # mask parts of the model
        masking_handles = inject_attention_head_mask(test_lm_model, head_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is the same as the nullified model
        assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)


class TestInjectAttentionLayerMask:
    def _get_test_mask(self, model: PreTrainedModel, layers_to_mask: list[int]) -> torch.Tensor:
        layer_mask = torch.ones((model.config.num_hidden_layers,), device=model.device)
        for layer in layers_to_mask:
            layer_mask[layer] = 0
        return layer_mask

    @torch.no_grad()
    def test_differs_from_original(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        # set params
        layers_to_mask = [0]
        layer_mask = self._get_test_mask(test_lm_model, layers_to_mask)

        # get output of the original model
        original_last_hidden_state = test_lm_model(**random_batch)[0]

        # inject mask
        masking_handles = inject_attention_layer_mask(test_lm_model, layer_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is different from the original model
        assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

        # remove mask via handles
        for handle in masking_handles:
            handle.remove()
        no_handles_last_hidden_state = test_lm_model(**random_batch)[0]

        assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)

    @torch.no_grad()
    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        # set params
        layers_to_mask = [0]
        layer_mask = self._get_test_mask(test_lm_model, layers_to_mask)

        # nullify parts of the model
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_attention_layers(nullified_model, layers_to_mask)
        nullified_last_hidden_state = nullified_model(**random_batch)[0]

        # mask parts of the model
        masking_handles = inject_attention_layer_mask(test_lm_model, layer_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is the same as the nullified model
        assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)


class TestInjectFfnNeuronMask:
    def _get_test_mask(self, model: PreTrainedModel, neurons_to_mask: dict[int, list[int]]) -> torch.Tensor:
        neuron_mask = torch.ones((model.config.num_hidden_layers, model.config.intermediate_size), device=model.device)
        for layer, neurons in neurons_to_mask.items():
            neuron_mask[layer, neurons] = 0
        return neuron_mask

    @torch.no_grad()
    def test_differs_from_original(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        # set params
        neurons_to_mask = {0: [0, 10, *range(50, 60), 127]}
        neuron_mask = self._get_test_mask(test_lm_model, neurons_to_mask)

        # get output of the original model
        original_last_hidden_state = test_lm_model(**random_batch)[0]

        # inject mask
        masking_handles = inject_ffn_neuron_mask(test_lm_model, neuron_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is different from the original model
        assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

        # remove mask via handles
        for handle in masking_handles:
            handle.remove()
        no_handles_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the model with removed handles is the same as the original model
        assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)

    @torch.no_grad()
    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        # set params
        neurons_to_mask = {0: [0, 10, *range(50, 60), 127]}
        neuron_mask = self._get_test_mask(test_lm_model, neurons_to_mask)

        # nullify parts of the model
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_ffn_neurons(nullified_model, neurons_to_mask)
        nullified_last_hidden_state = nullified_model(**random_batch)[0]

        # mask parts of the model
        _ = inject_ffn_neuron_mask(test_lm_model, neuron_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is the same as the nullified model
        assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)


class TestInjectFfnLayerMask:

    def _get_test_mask(self, model: PreTrainedModel, layers_to_mask: list[int]) -> torch.Tensor:
        layer_mask = torch.ones((model.config.num_hidden_layers,), device=model.device)
        for layer in layers_to_mask:
            layer_mask[layer] = 0
        return layer_mask

    @torch.no_grad()
    def test_differs_from_original(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        # set params
        layers_to_mask = [0]
        layer_mask = self._get_test_mask(test_lm_model, layers_to_mask)

        # get output of the original model
        original_last_hidden_state = test_lm_model(**random_batch)[0]

        # inject mask
        masking_handles = inject_ffn_layer_mask(test_lm_model, layer_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is different from the original model
        assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

        # remove mask via handles
        for handle in masking_handles:
            handle.remove()
        no_handles_last_hidden_state = test_lm_model(**random_batch)[0]

        assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)

    @torch.no_grad()
    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        # set params
        layers_to_mask = [0]
        layer_mask = self._get_test_mask(test_lm_model, layers_to_mask)

        # nullify parts of the model
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_ffn_layers(nullified_model, layers_to_mask)
        nullified_last_hidden_state = nullified_model(**random_batch)[0]

        # mask parts of the model
        _ = inject_ffn_layer_mask(test_lm_model, layer_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is the same as the nullified model
        assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)


class TestInjectHiddenStateMask:

    def _get_test_mask(self, model: PreTrainedModel, hidden_state_to_mask: list[int]) -> torch.Tensor:
        hidden_state_mask = torch.ones((model.config.hidden_size,), device=model.device)
        for index in hidden_state_to_mask:
            hidden_state_mask[index] = 0
        return hidden_state_mask

    @torch.no_grad()
    def test_differs_from_original(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        if isinstance(test_lm_model, BertForMaskedLM):
            pytest.xfail("BertForMaskedLM is not supported for hidden state masking yet")

        # set params
        hidden_state_to_mask = [0, 10, *range(50, 60)]
        hidden_state_mask = self._get_test_mask(test_lm_model, hidden_state_to_mask)

        # get output of the original model
        original_last_hidden_state = test_lm_model(**random_batch)[0]

        # inject mask
        masking_handles = inject_hidden_state_mask(test_lm_model, hidden_state_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is different from the original model
        assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

        # remove mask via handles
        for handle in masking_handles:
            handle.remove()
        no_handles_last_hidden_state = test_lm_model(**random_batch)[0]

        assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)

    @torch.no_grad()
    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_batch) -> None:
        if isinstance(test_lm_model, BertForMaskedLM):
            pytest.xfail("BertForMaskedLM is not supported for hidden state masking yet")

        # set params
        hidden_state_to_mask = [0, 10, *range(50, 60)]
        hidden_state_mask = self._get_test_mask(test_lm_model, hidden_state_to_mask)

        # nullify parts of the model
        nullified_model = copy.deepcopy(test_lm_model)
        nullify_hidden_states(nullified_model, hidden_state_to_mask)
        nullified_last_hidden_state = nullified_model(**random_batch)[0]

        # mask parts of the model
        _ = inject_hidden_state_mask(test_lm_model, hidden_state_mask)
        masked_last_hidden_state = test_lm_model(**random_batch)[0]

        # check the output of the masked model is the same as the nullified model
        assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)
