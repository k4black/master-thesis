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
        self, collector: Callable, bert_clf_test_model: PreTrainedModel, random_dataloader: DataLoader
    ) -> None:
        result = collector(bert_clf_test_model, random_dataloader)
        assert isinstance(result, ComponentsInfo)
        for field_name, value in result._asdict().items():
            assert value.shape[0] in [0, 2, 8], field_name  # batched or full


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
