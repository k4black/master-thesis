from typing import Callable

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

from adaptive_pruning.importance import (
    ComponentsImportance,
    ComponentsInfo,
    collect_activations,
    collect_mask_gradients,
    collect_random_numbers,
    collect_weight_magnitudes,
    info_to_entropy,
    info_to_fisher,
    info_to_max,
    info_to_mean,
    select_attention_heads,
    select_to_prune_attention_layers,
    select_to_prune_ffn_layers,
    select_to_prune_ffn_neurons,
)


@pytest.fixture
def random_info() -> ComponentsInfo:
    return ComponentsInfo(*[torch.rand(10, 5) for _ in range(6)])


class TestCollect:
    @pytest.mark.parametrize(
        "collector",
        [
            collect_mask_gradients,
            collect_activations,
            collect_weight_magnitudes,
            collect_random_numbers,
        ],
    )
    def test_all_collectors_random_input(
        self, collector: Callable, test_lm_model_llama: PreTrainedModel, random_lm_dataloader: DataLoader
    ) -> None:
        result = collector(test_lm_model_llama, random_lm_dataloader)
        assert isinstance(result, ComponentsInfo)
        for field_name, value in result._asdict().items():
            assert value.shape[0] in [1, 2, 8], field_name  # batched or full


class TestInfoTo:
    @pytest.mark.parametrize(
        "info_processor",
        [
            info_to_mean,
            info_to_max,
            info_to_fisher,
            info_to_entropy,
        ],
    )
    def test_all_processors_random_input(self, info_processor: Callable, random_info: ComponentsInfo) -> None:
        result = info_processor(random_info)
        assert isinstance(result, ComponentsImportance)
        for field_name, value in result._asdict().items():
            assert value.shape == (5,), field_name
            # assert torch.all(r >= 0)


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
        selected_heads = select_attention_heads(importance_scores, percent_heads_to_prune)
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
        selected_heads = select_attention_heads(importance_scores, percent_heads_to_prune)
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
        selected_heads = select_attention_heads(importance_scores, percent_heads_to_prune, uniform_among_layers=True)
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
