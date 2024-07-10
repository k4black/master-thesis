import copy

import pytest
import torch
from transformers import BertForMaskedLM, PreTrainedModel

from adaptive_pruning.nullify import (
    nullify_attention_heads,
    nullify_attention_layers,
    nullify_ffn_layers,
    nullify_ffn_neurons,
    nullify_hidden_state,
)
from adaptive_pruning.pruning import (
    prune_attention_heads,
    prune_attention_layers,
    prune_ffn_layers,
    prune_ffn_neurons,
    prune_hidden_state,
    select_to_prune_attention_heads,
    select_to_prune_attention_layers,
    select_to_prune_ffn_layers,
    select_to_prune_ffn_neurons,
)
from adaptive_pruning.utils import count_flops_macs_params, count_total_parameters


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
        bert_lm_test_model: PreTrainedModel,
        heads_to_prune: dict[int, list[int]],
        expected_heads: list[int],
    ) -> None:
        self._assert_attention_heads_number_by_layer(bert_lm_test_model, [4, 4])

        prune_attention_heads(bert_lm_test_model, heads_to_prune)

        self._assert_attention_heads_number_by_layer(bert_lm_test_model, expected_heads)

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
        llama_lm_test_model: PreTrainedModel,
        heads_to_prune: dict[int, list[int]],
        expected_heads: list[int],
    ) -> None:
        self._assert_attention_heads_number_by_layer(llama_lm_test_model, [4, 4])

        prune_attention_heads(llama_lm_test_model, heads_to_prune)

        self._assert_attention_heads_number_by_layer(llama_lm_test_model, expected_heads)

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

    def test_requires_grad_bert(self, bert_test_model: PreTrainedModel) -> None:
        # set params
        heads_to_prune = {0: [0]}

        # save old requires_grad
        old_requires_grad_query = bert_test_model.encoder.layer[0].attention.self.query.weight.requires_grad
        old_requires_grad_bias = bert_test_model.encoder.layer[0].attention.self.query.bias.requires_grad

        # prune
        prune_attention_heads(bert_test_model, heads_to_prune)

        # check that the model requires grad
        assert bert_test_model.encoder.layer[0].attention.self.query.weight.requires_grad == old_requires_grad_query
        assert bert_test_model.encoder.layer[0].attention.self.query.bias.requires_grad == old_requires_grad_bias

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:
        if isinstance(test_lm_model, BertForMaskedLM):
            pytest.xfail("BertForMaskedLM does not support nullify_attention_heads for now")

        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = test_lm_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # set params
            head_index = 0
            heads_to_prune = {0: [head_index]}

            # nullify first head of the first layer
            nullified_model = copy.deepcopy(test_lm_model)
            nullify_attention_heads(nullified_model, heads_to_prune)
            # get output of the nullified model
            nullified_last_hidden_state = nullified_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # check the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # prune the first head of the first layer
            prune_attention_heads(test_lm_model, heads_to_prune)
            # get output of the pruned model
            pruned_last_hidden_state = test_lm_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # check the output of the pruned model is different from the original model
            assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

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

    def test_requires_grad_bert(self, bert_test_model: PreTrainedModel) -> None:
        # set params
        layers_to_prune = [0]

        # save old requires_grad
        old_requires_grad = bert_test_model.encoder.layer[0].attention.output.dense.weight.requires_grad

        # prune
        prune_attention_layers(bert_test_model, layers_to_prune)

        # check that the model requires grad
        assert bert_test_model.encoder.layer[0].attention.output.dense.weight.requires_grad == old_requires_grad

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:

        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = test_lm_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # set params
            layer_index = 0
            layers_to_prune = [layer_index]

            # nullify first layer
            nullified_model = copy.deepcopy(test_lm_model)
            nullify_attention_layers(nullified_model, layers_to_prune)
            # get output of the nullified model
            nullified_last_hidden_state = nullified_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # check the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # prune the first layer
            prune_attention_layers(test_lm_model, layers_to_prune)
            # get output of the pruned model
            pruned_last_hidden_state = test_lm_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # check the output of the pruned model is different from the original model
            assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

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

    def test_requires_grad_bert(self, bert_test_model: PreTrainedModel) -> None:
        # set params
        neurons_to_prune = {0: [0, 10, *range(50, 60), 127]}

        # save old requires_grad
        old_requires_grad_intermediate = bert_test_model.encoder.layer[0].intermediate.dense.weight.requires_grad
        old_requires_grad_output = bert_test_model.encoder.layer[0].output.dense.weight.requires_grad

        # prune
        prune_ffn_neurons(bert_test_model, neurons_to_prune)

        # check that the model requires grad
        assert (
            bert_test_model.encoder.layer[0].intermediate.dense.weight.requires_grad == old_requires_grad_intermediate
        )
        assert bert_test_model.encoder.layer[0].output.dense.weight.requires_grad == old_requires_grad_output

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:

        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = test_lm_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # set params
            neurons_to_prune = {0: [0, 10, *range(50, 60), 127]}

            # nullify first neuron of the first layer
            nullified_model = copy.deepcopy(test_lm_model)
            nullify_ffn_neurons(nullified_model, neurons_to_prune)
            # get output of the nullified model
            nullified_last_hidden_state = nullified_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # check the output of the nullified model is different from the original model
            assert original_last_hidden_state.shape == nullified_last_hidden_state.shape
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state, atol=1e-5)

            # prune the first neuron of the first layer
            prune_ffn_neurons(test_lm_model, neurons_to_prune)
            # get output of the pruned model
            pruned_last_hidden_state = test_lm_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # check the output of the pruned model is different from the original model
            assert original_last_hidden_state.shape == pruned_last_hidden_state.shape
            assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state, atol=1e-5)

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

    def test_requires_grad_bert(self, bert_test_model: PreTrainedModel) -> None:
        # set params
        layers_to_prune = [0]

        # save old requires_grad
        old_requires_grad_intermediate = bert_test_model.encoder.layer[0].intermediate.dense.weight.requires_grad
        old_requires_grad_output = bert_test_model.encoder.layer[0].output.dense.weight.requires_grad

        # prune
        prune_ffn_layers(bert_test_model, layers_to_prune)

        # check that the model requires grad
        assert (
            bert_test_model.encoder.layer[0].intermediate.dense.weight.requires_grad == old_requires_grad_intermediate
        )
        assert bert_test_model.encoder.layer[0].output.dense.weight.requires_grad == old_requires_grad_output

    def test_same_as_nullify(self, test_lm_model: PreTrainedModel, random_input_batch) -> None:

        with torch.no_grad():
            # get output of the original model
            original_last_hidden_state = test_lm_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # set params
            layers_to_prune = [0]

            # nullify first layer
            nullified_model = copy.deepcopy(test_lm_model)
            nullify_ffn_layers(nullified_model, layers_to_prune)
            # get output of the nullified model
            nullified_last_hidden_state = nullified_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # check the output of the nullified model is different from the original model
            assert not torch.allclose(nullified_last_hidden_state, original_last_hidden_state)

            # prune the first layer
            prune_ffn_layers(test_lm_model, layers_to_prune)
            # get output of the pruned model
            pruned_last_hidden_state = test_lm_model(
                random_input_batch["input_ids"], random_input_batch["attention_mask"]
            )[0]

            # check the output of the pruned model is different from the original model
            assert not torch.allclose(pruned_last_hidden_state, original_last_hidden_state)

            # check the output of the pruned model is the same as the nullified model
            assert torch.allclose(pruned_last_hidden_state, nullified_last_hidden_state)


class TestPruneHiddenState:
    @staticmethod
    def _assert_hidden_state_neurons_number(model: PreTrainedModel, expected_hidden_state: int) -> None:
        model, architecture = model.base_model, model.config.model_type

        assert model.config.hidden_size == expected_hidden_state

        if architecture == "bert":
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
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    @pytest.mark.parametrize(
        "neurons_to_prune, expected_hidden_state",
        [
            ([], 64),
            ([0], 63),
            ([0, 1], 62),
            ([0, 1, 2], 61),
            (list(range(60)), 4),
            ([0, 1, 51, *range(10, 20)], 51),
        ],
    )
    def test_num_neurons(
        self,
        bert_test_model: PreTrainedModel,
        neurons_to_prune: list[int],
        expected_hidden_state: int,
    ) -> None:
        self._assert_hidden_state_neurons_number(bert_test_model, 64)

        prune_hidden_state(bert_test_model, neurons_to_prune)

        self._assert_hidden_state_neurons_number(bert_test_model, expected_hidden_state)

    def test_pass_with_correct_dim(self, bert_test_model: PreTrainedModel, random_input_batch) -> None:
        # get output of the original model
        original_last_hidden_state = bert_test_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]
        assert original_last_hidden_state.shape == (
            *random_input_batch["input_ids"].shape,
            bert_test_model.config.hidden_size,
        )

        # set params
        neurons_to_prune = [0, 1, 63, *range(10, 20)]

        # prune the hidden state
        prune_hidden_state(bert_test_model, neurons_to_prune)
        # get output of the pruned model
        pruned_last_hidden_state = bert_test_model(
            random_input_batch["input_ids"], random_input_batch["attention_mask"]
        )[0]
        assert pruned_last_hidden_state.shape == (*random_input_batch["input_ids"].shape, 51)


class TestSelectAttentionHeads:
    @pytest.mark.parametrize(
        "importance_scores, percent_heads_to_prune, expected_heads_to_prune",
        [
            # 1 layer, 2 heads, 50% to prune
            (torch.tensor([[0.6, 0.3]]), 0.5, {0: [1]}),
            # negative importance
            (torch.tensor([[-0.1, -0.02]]), 0.5, {0: [0]}),
            # 2 layers, 2 heads, 50% to prune
            (torch.tensor([[0.6, 0.1], [0.5, 0.2]]), 0.5, {0: [1], 1: [1]}),
            # 2 layers, 4 heads, 50% to prune
            (torch.tensor([[0.6, 0.3, 0.1, 0.2], [0.5, 0.2, 0.1, 0.4]]), 0.5, {0: [2, 3], 1: [2, 1]}),
            # 2 layers, 4 heads, 75% to prune
            (torch.tensor([[0.6, 0.3, 0.1, 0.2], [0.5, 0.2, 0.3, 0.4]]), 0.75, {0: [2, 3, 1], 1: [1, 2, 3]}),
            # 2 layers, 4 heads, 25% to prune
            (torch.tensor([[0.6, 0.3, 0.1, 0.3], [0.5, 0.2, 0.3, 0.4]]), 0.25, {0: [2], 1: [1]}),
            # 2 layers, 4 heads, 25% to prune - one layer less important
            (torch.tensor([[0.6, 0.3, 0.1, -1.0], [0.5, 0.2, 0.3, 0.4]]), 0.25, {0: [3, 2]}),
            # 2 layers, 4 heads, 50% to prune - one layer less important
            (torch.tensor([[0.6, 0.1, 0.1, 0.1], [0.5, 0.2, 0.3, 0.4]]), 0.5, {0: [1, 2, 3], 1: [1]}),
        ],
    )
    def test_simple_cases_global(
        self,
        importance_scores: torch.Tensor,
        percent_heads_to_prune: float,
        expected_heads_to_prune: dict[int, list[int]],
    ) -> None:
        selected_heads = select_to_prune_attention_heads(importance_scores, percent_heads_to_prune)
        assert selected_heads == expected_heads_to_prune

    @pytest.mark.parametrize(
        "importance_scores, percent_heads_to_prune, expected_heads_to_prune",
        [
            # 2 layers, 2 heads, 100% to prune - but keep 1 head per layer
            (torch.tensor([[0.6, 0.3], [0.1, 0.2]]), 1.0, {0: [1], 1: [0]}),
            # 2 layers, 2 heads, 50% to prune - one layer less important, but keep 1 head per layer
            (torch.tensor([[0.6, 0.3], [0.1, 0.2]]), 0.5, {1: [0]}),
        ],
    )
    def test_keep_at_least_one_head_global(
        self,
        importance_scores: torch.Tensor,
        percent_heads_to_prune: float,
        expected_heads_to_prune: dict[int, list[int]],
    ) -> None:
        selected_heads = select_to_prune_attention_heads(importance_scores, percent_heads_to_prune)
        assert selected_heads == expected_heads_to_prune

    @pytest.mark.parametrize(
        "importance_scores, percent_heads_to_prune, expected_heads_to_prune",
        [
            # 1 layer, 2 heads, 50% to prune
            (torch.tensor([[0.6, 0.3]]), 0.5, {0: [1]}),
            # 2 layers, 2 heads, 50% to prune
            (torch.tensor([[0.6, 0.1], [0.5, 0.2]]), 0.5, {0: [1], 1: [1]}),
            # 2 layers, 4 heads, 25% to prune - one layer less important
            (torch.tensor([[0.6, 0.3, 0.1, -1.0], [0.5, 0.2, 0.3, 0.4]]), 0.25, {0: [3], 1: [1]}),
            # 2 layers, 4 heads, 50% to prune - one layer less important
            (torch.tensor([[0.6, 0.1, 0.2, 0.1], [0.5, 0.2, 0.3, 0.4]]), 0.5, {0: [1, 3], 1: [1, 2]}),
        ],
    )
    def test_simple_cases_uniform(
        self,
        importance_scores: torch.Tensor,
        percent_heads_to_prune: float,
        expected_heads_to_prune: dict[int, list[int]],
    ) -> None:
        selected_heads = select_to_prune_attention_heads(
            importance_scores, percent_heads_to_prune, uniform_among_layers=True
        )
        assert selected_heads == expected_heads_to_prune


class TestSelectAttentionLayers:
    @pytest.mark.parametrize(
        "importance_scores, percent_layers_to_prune, expected_layers_to_prune",
        [
            # 2 layers, 50% to prune
            (torch.tensor([0.6, 0.3]), 0.5, [1]),
            # 2 layers, 100% to prune
            (torch.tensor([0.6, 0.3]), 1.0, [1, 0]),
            # 2 layers, 25% to prune
            (torch.tensor([0.6, 0.3]), 0.25, []),
            # 4 layers, 25% to prune
            (torch.tensor([0.6, 0.3, -1.0, -0.02]), 0.25, [2]),
            # 4 layers, 75% to prune
            (torch.tensor([0.6, 0.3, -1.0, -0.02]), 0.75, [2, 3, 1]),
        ],
    )
    def test_simple_cases(
        self, importance_scores: torch.Tensor, percent_layers_to_prune: float, expected_layers_to_prune: list[int]
    ) -> None:
        selected_layers = select_to_prune_attention_layers(importance_scores, percent_layers_to_prune)
        assert selected_layers == expected_layers_to_prune


class TestSelectFnnNeurons:
    @pytest.mark.parametrize(
        "importance_scores, percent_neurons_to_prune, expected_neurons_to_prune",
        [
            # 1 layer, 4 neurons, 50% to prune
            (torch.tensor([[0.6, 0.3, 0.1, 0.2]]), 0.5, {0: [2, 3]}),
            # 1 layer, 4 neurons, 75% to prune
            (torch.tensor([[0.6, 0.3, 0.1, 0.2]]), 0.75, {0: [2, 3, 1]}),
            # 2 layers, 4 neurons, 50% to prune
            (torch.tensor([[0.6, 0.3, 0.1, 0.2], [0.5, 0.2, 0.1, 0.4]]), 0.5, {0: [2, 3], 1: [2, 1]}),
            # 2 layers, 4 neurons, 50% to prune - one layer less important
            (torch.tensor([[0.6, 0.3, 0.1, 0.2], [0.5, 0.5, 0.5, 0.1]]), 0.5, {0: [2, 3, 1], 1: [3]}),
        ],
    )
    def test_simple_cases_global(
        self,
        importance_scores: torch.Tensor,
        percent_neurons_to_prune: float,
        expected_neurons_to_prune: dict[int, list[int]],
    ) -> None:
        selected_neurons = select_to_prune_ffn_neurons(importance_scores, percent_neurons_to_prune)
        assert selected_neurons == expected_neurons_to_prune

    @pytest.mark.parametrize(
        "importance_scores, percent_neurons_to_prune, expected_neurons_to_prune",
        [
            # 1 layer, 4 neurons, 50% to prune
            (torch.tensor([[0.6, 0.3, 0.1, 0.2]]), 0.5, {0: [2, 3]}),
            # 2 layers, 4 neurons, 50% to prune - one layer less important
            (torch.tensor([[0.6, 0.3, 0.1, 0.2], [0.5, 0.4, 0.5, 0.1]]), 0.5, {0: [2, 3], 1: [3, 1]}),
        ],
    )
    def test_simple_cases_uniform(
        self,
        importance_scores: torch.Tensor,
        percent_neurons_to_prune: float,
        expected_neurons_to_prune: dict[int, list[int]],
    ) -> None:
        selected_neurons = select_to_prune_ffn_neurons(
            importance_scores, percent_neurons_to_prune, uniform_among_layers=True
        )
        assert selected_neurons == expected_neurons_to_prune


class TestSelectFnnLayers:
    @pytest.mark.parametrize(
        "importance_scores, percent_layers_to_prune, expected_layers_to_prune",
        [
            # 2 layers, 50% to prune
            (torch.tensor([0.6, 0.3]), 0.5, [1]),
            # 2 layers, 100% to prune
            (torch.tensor([0.6, 0.3]), 1.0, [1, 0]),
            # 2 layers, 25% to prune
            (torch.tensor([0.6, 0.3]), 0.25, []),
            # 4 layers, 25% to prune
            (torch.tensor([0.6, 0.3, -1.0, -0.02]), 0.25, [2]),
            # 4 layers, 75% to prune
            (torch.tensor([0.6, 0.3, -1.0, -0.02]), 0.75, [2, 3, 1]),
        ],
    )
    def test_simple_cases(
        self, importance_scores: torch.Tensor, percent_layers_to_prune: float, expected_layers_to_prune: list[int]
    ) -> None:
        selected_layers = select_to_prune_ffn_layers(importance_scores, percent_layers_to_prune)
        assert selected_layers == expected_layers_to_prune
