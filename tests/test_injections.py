import copy

import torch
from transformers import PreTrainedModel

from adaptive_pruning.injections import (
    inject_attention_head_mask, inject_attention_layer_mask, inject_ffn_neuron_mask, inject_ffn_layer_mask
)
from adaptive_pruning.nullify import (
    nullify_attention_heads, nullify_attention_layers, nullify_ffn_neurons, nullify_ffn_layers, nullify_hidden_state
)


class TestInjectAttentionHeadMask:

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:
        
        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # set params
            heads_to_mask = {0: [0]}
            head_mask = torch.ones((2, 4), device=test_lm_model.device)
            for layer, heads in heads_to_mask.items():
                head_mask[layer, heads] = 0

            # nullify parts of the model
            nullified_model = copy.deepcopy(test_lm_model)
            nullify_attention_heads(nullified_model, heads_to_mask)
            nullified_last_hidden_state = nullified_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # mask parts of the model
            masking_handles = inject_attention_head_mask(test_lm_model, head_mask)
            masked_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the masked model is different from the original model
            assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

            # check the output of the masked model is the same as the nullified model
            assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)

            # remove mask via handles
            for handle in masking_handles:
                handle.remove()
            no_handles_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the model with removed handles is the same as the original model
            assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)


class TestInjectAttentionLayerMask:

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:
        
        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # set params
            layers_to_mask = [0]
            layer_mask = torch.ones((2,), device=test_lm_model.device)
            for layer in layers_to_mask:
                layer_mask[layer] = 0

            # nullify parts of the model
            nullified_model = copy.deepcopy(test_lm_model)
            nullify_attention_layers(nullified_model, layers_to_mask)
            nullified_last_hidden_state = nullified_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # mask parts of the model
            masking_handles = inject_attention_layer_mask(test_lm_model, layer_mask)
            masked_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the masked model is different from the original model
            assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

            # check the output of the masked model is the same as the nullified model
            assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)

            # remove mask via handles
            for handle in masking_handles:
                handle.remove()
            no_handles_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the model with removed handles is the same as the original model
            assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)


class TestInjectFfnNeuronMask:

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:
        
        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # set params
            neurons_to_mask = {0: [0, 10, *range(50, 60), 127]}
            neuron_mask = torch.ones((2, test_lm_model.config.intermediate_size), device=test_lm_model.device)
            for layer, neurons in neurons_to_mask.items():
                neuron_mask[layer, neurons] = 0

            # nullify parts of the model
            nullified_model = copy.deepcopy(test_lm_model)
            nullify_ffn_neurons(nullified_model, neurons_to_mask)
            nullified_last_hidden_state = nullified_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # mask parts of the model
            masking_handles = inject_ffn_neuron_mask(test_lm_model, neuron_mask)
            masked_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the masked model is different from the original model
            assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

            # check the output of the masked model is the same as the nullified model
            assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)

            # remove mask via handles
            for handle in masking_handles:
                handle.remove()
            no_handles_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the model with removed handles is the same as the original model
            assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)


class TestInjectFfnLayerMask:

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:
        
        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # set params
            layers_to_mask = [0]
            layer_mask = torch.ones((2,), device=test_lm_model.device)
            for layer in layers_to_mask:
                layer_mask[layer] = 0

            # nullify parts of the model
            nullified_model = copy.deepcopy(test_lm_model)
            nullify_ffn_layers(nullified_model, layers_to_mask)
            nullified_last_hidden_state = nullified_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # mask parts of the model
            masking_handles = inject_ffn_layer_mask(test_lm_model, layer_mask)
            masked_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the masked model is different from the original model
            assert not torch.allclose(masked_last_hidden_state, original_last_hidden_state)

            # check the output of the masked model is the same as the nullified model
            assert torch.allclose(masked_last_hidden_state, nullified_last_hidden_state)

            # remove mask via handles
            for handle in masking_handles:
                handle.remove()
            no_handles_last_hidden_state = test_lm_model(random_input_batch["input_ids"], random_input_batch["attention_mask"])[0]

            # check the output of the model with removed handles is the same as the original model
            assert torch.allclose(no_handles_last_hidden_state, original_last_hidden_state)

