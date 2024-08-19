from typing import Callable

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import PretrainedConfig, PreTrainedModel

from adaptive_pruning.importance import (
    ComponentsImportance,
    ComponentsInfo,
    ComponentsToPrune,
    collect_activations,
    collect_mask_gradients,
    collect_random_numbers,
    collect_weight_magnitudes,
    info_to_entropy,
    info_to_fisher,
    info_to_max,
    info_to_mean,
    select_to_prune_attention_heads,
    select_to_prune_attention_layers,
    select_to_prune_ffn_layers,
    select_to_prune_ffn_neurons, get_components_ratios,
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
        result: ComponentsInfo = collector(test_lm_model_llama, random_lm_dataloader)
        assert isinstance(result, ComponentsInfo)
        for field_name, value in result._asdict().items():
            assert value.shape[0] in [1, 2, 8], field_name  # batched or full

        # check samples (first dimension) are different
        if result[0].shape[0] > 1:
            assert not torch.allclose(result.attention_heads_info[0], result.attention_heads_info[1])
            assert not torch.allclose(result.attention_layers_info[0], result.attention_layers_info[1])
            assert not torch.allclose(result.ffn_neurons_info[0], result.ffn_neurons_info[1])
            assert not torch.allclose(result.ffn_layers_info[0], result.ffn_layers_info[1])
            assert not torch.allclose(result.hidden_states_info[0], result.hidden_states_info[1])
            assert not torch.allclose(result.meta_info[0], result.meta_info[1])


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

    import pytest

    from adaptive_pruning.importance import ComponentsImportance, ComponentsInfo

    @pytest.fixture
    def example_components_info(self) -> ComponentsInfo:
        return ComponentsInfo(
            attention_heads_info=torch.rand(10, 5, 8),  # 10 samples, 5 layers, 8 heads
            attention_layers_info=torch.rand(10, 5),  # 10 samples, 5 layers
            ffn_neurons_info=torch.rand(10, 5, 128),  # 10 samples, 5 layers, 128 neurons
            ffn_layers_info=torch.rand(10, 5),  # 10 samples, 5 layers
            hidden_states_info=torch.rand(10, 512),  # 10 samples, 512 hidden states
            meta_info=torch.rand(10, 5),  # 10 samples, 5 meta info
        )

    @pytest.mark.parametrize("how_to_average", ["fisher_info", "mean", "max", "entropy", "minus_entropy"])
    def test_from_info(self, example_components_info: ComponentsInfo, how_to_average: str) -> None:
        components_importance = ComponentsImportance.from_info(example_components_info, how_to_average)  # type: ignore

        # Check if the returned object is an instance of ComponentsImportance
        assert isinstance(components_importance, ComponentsImportance)

        # Check the shapes of the tensors in the returned ComponentsImportance object
        assert components_importance.attention_heads_importance.shape == (5, 8)
        assert components_importance.attention_layers_importance.shape == (5,)
        assert components_importance.ffn_neurons_importance.shape == (5, 128)
        assert components_importance.ffn_layers_importance.shape == (5,)
        assert components_importance.hidden_states_importance.shape == (512,)
        assert components_importance.meta_importance.shape == (5,)


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
        selected_heads = select_to_prune_attention_heads(
            importance_scores, percent_heads_to_prune, keep_at_least_one_head=True
        )
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
            importance_scores,
            percent_heads_to_prune,
            uniform_among_layers=True,
        )
        assert selected_heads == expected_heads_to_prune

    @pytest.mark.parametrize("round_to_heads", [1, 2, 3, 4, 5, 6, 7, 16])
    @pytest.mark.parametrize("is_uniform", [False, True])
    def test_rounding(self, round_to_heads: int, is_uniform: bool) -> None:
        torch.manual_seed(42)
        importance_scores = torch.rand(10, 100)  # 10 layers, 100 heads
        percent_heads_to_prune = 0.231
        selected_heads = select_to_prune_attention_heads(
            importance_scores,
            percent_heads_to_prune,
            round_to_heads=round_to_heads,
            uniform_among_layers=is_uniform,
        )

        for layer, heads in selected_heads.items():
            assert len(heads) % round_to_heads == 0, f"Layer {layer} has {len(heads)} heads, expected x{round_to_heads}"


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


class TestSelectFfnNeurons:
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

    @pytest.mark.parametrize("round_to_neurons", [1, 2, 3, 4, 5, 6, 7, 16, 32, 64, 128])
    @pytest.mark.parametrize("is_uniform", [False, True])
    def test_rounding(self, round_to_neurons: int, is_uniform: bool) -> None:
        torch.manual_seed(42)
        importance_scores = torch.rand(10, 1000)

        percent_neurons_to_prune = 0.231
        selected_neurons = select_to_prune_ffn_neurons(
            importance_scores,
            percent_neurons_to_prune,
            round_to=round_to_neurons,
            uniform_among_layers=is_uniform,
        )

        for layer, neurons in selected_neurons.items():
            assert (
                len(neurons) % round_to_neurons == 0
            ), f"Layer {layer} has {len(neurons)} neurons, expected x{round_to_neurons}"


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


class TestSelectToPrune:

    @pytest.fixture
    def mock_components_importance(self) -> ComponentsImportance:
        # Mock data for ComponentsImportance
        return ComponentsImportance(
            attention_heads_importance=torch.rand(8, 12),
            attention_layers_importance=torch.rand(8),
            ffn_neurons_importance=torch.rand(8, 768),
            ffn_layers_importance=torch.rand(8),
            hidden_states_importance=torch.rand(768),
            meta_importance=torch.rand(5),
        )

    @pytest.fixture
    def mock_config(self) -> PretrainedConfig:
        # Mock PretrainedConfig with specific attributes
        config = PretrainedConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=8,
            intermediate_size=3072,
            num_key_value_heads=4,
        )
        return config

    @pytest.mark.parametrize(
        "pruning_components",
        [["attn_heads"], ["attn_layers"], ["ffn_neurons"], ["ffn_layers"], ["hidden_states"]],
    )
    def test_from_importance_uniform_pruning(
        self,
        mock_components_importance: ComponentsImportance,
        mock_config: PretrainedConfig,
        pruning_components: list[str],
    ):
        # Test uniform pruning across layers
        components_to_prune = ComponentsToPrune.from_importance(
            components_importance=mock_components_importance,
            pruning_ratio_target=0.5,
            pruning_components=pruning_components,
            round_to=1,
            is_uniform=True,
            how_to_overlap="fixed",
            config=mock_config,
        )
        # Assertions to verify the correct components are selected for pruning
        if "attn_heads" in pruning_components:
            assert len(components_to_prune.attention_heads_to_prune) == 8
            for layer, heads in components_to_prune.attention_heads_to_prune.items():
                assert len(heads) == 6
        else:
            assert not components_to_prune.attention_heads_to_prune
        if "attn_layers" in pruning_components:
            assert len(components_to_prune.attention_layers_to_prune) == 4
        else:
            assert not components_to_prune.attention_layers_to_prune
        if "ffn_neurons" in pruning_components:
            assert len(components_to_prune.ffn_neurons_to_prune) == 8
            for layer, neurons in components_to_prune.ffn_neurons_to_prune.items():
                assert len(neurons) == 384
        else:
            assert not components_to_prune.ffn_neurons_to_prune
        if "ffn_layers" in pruning_components:
            assert len(components_to_prune.ffn_layers_to_prune) == 4
        else:
            assert not components_to_prune.ffn_layers_to_prune
        if "hidden_states" in pruning_components:
            assert len(components_to_prune.hidden_states_to_prune) == 384
        else:
            assert not components_to_prune.hidden_states_to_prune

    @pytest.mark.parametrize(
        "pruning_components",
        [["attn_heads"], ["attn_layers"], ["ffn_neurons"], ["ffn_layers"], ["hidden_states"]],
    )
    def test_from_importance_with_skip_pruned_components(
        self,
        mock_components_importance: ComponentsImportance,
        mock_config: PretrainedConfig,
        pruning_components: list[str],
    ):
        # Test behavior when skip_pruned_components is provided
        skip_pruned_components = ComponentsToPrune(
            attention_heads_to_prune={0: [1, 2], 1: [3, 4]},
            attention_layers_to_prune=[1],
            ffn_neurons_to_prune={2: [100, 200]},
            ffn_layers_to_prune=[2],
            hidden_states_to_prune=[10, 20, 30],
        )
        components_to_prune = ComponentsToPrune.from_importance(
            components_importance=mock_components_importance,
            pruning_ratio_target=0.9,
            pruning_components=pruning_components,
            round_to=1,
            is_uniform=False,
            how_to_overlap="fixed",
            config=mock_config,
            already_pruned_components=skip_pruned_components,
        )
        # Assertions to verify that skipped components are not pruned again
        for i in [1, 2]:
            assert i not in components_to_prune.attention_heads_to_prune.get(0, [])
        for i in [3, 4]:
            assert i not in components_to_prune.attention_heads_to_prune.get(1, [])
        assert 1 not in components_to_prune.attention_layers_to_prune
        for i in [100, 200]:
            assert i not in components_to_prune.ffn_neurons_to_prune.get(2, [])
        assert 2 not in components_to_prune.ffn_layers_to_prune
        for i in [10, 20, 30]:
            assert i not in components_to_prune.hidden_states_to_prune


# test get_components_ratios
class TestPruningRatiosSelection:
    POSSIBLE_COMPONENTS = ["attn_heads", "attn_layers", "ffn_neurons", "ffn_layers", "hidden_states"]

    @pytest.mark.parametrize("pruning_ratio", [0.1, 0.2, 0.5, 0.8])
    @pytest.mark.parametrize(
        "pruning_components",
        [
            ["attn_heads"],
            ["attn_layers"],
            ["ffn_neurons"],
            ["ffn_layers"],
            ["hidden_states"],
            ["attn_heads", "attn_layers", "ffn_neurons", "ffn_layers", "hidden_states"],
        ],
    )
    def test_fixed_overlap(self, pruning_ratio: float, pruning_components: list[str]) -> None:
        attn_heads_r, attn_layers_r, ffn_neurons_r, ffn_layers_r, hidden_states_r = get_components_ratios(
            pruning_ratio, pruning_components, how_to_overlap="fixed"
        )
        predicted = {
            "attn_heads": attn_heads_r,
            "attn_layers": attn_layers_r,
            "ffn_neurons": ffn_neurons_r,
            "ffn_layers": ffn_layers_r,
            "hidden_states": hidden_states_r,
        }

        for component in self.POSSIBLE_COMPONENTS:
            if component in pruning_components:
                assert predicted[component] == pruning_ratio, f"Component {component} should be pruned at {pruning_ratio}"
            else:
                assert predicted[component] == 0.0, f"Component {component} should not be pruned"
