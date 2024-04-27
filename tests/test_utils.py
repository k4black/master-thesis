import pytest
from transformers import PreTrainedModel
from torch import nn
from torchinfo import summary

from adaptive_pruning.utils import count_parameters, format_number


class TestCountParameters:

    @pytest.mark.parametrize(
        "module, expected",
        [
            (nn.Linear(10, 5, bias=False), 10*5),
            (nn.Linear(10, 5, bias=True), 10*5+5),
            (nn.Sequential(nn.Linear(10, 5, bias=False), nn.Linear(5, 4, bias=False)), 10*5+5*4),
            (nn.MultiheadAttention(10, 5, bias=False), 10*5*4*2),
        ],
    )
    def test_pytorch_module(self, module: nn.Module, expected: int) -> None:
        num_parameters = count_parameters(module)

        assert num_parameters == expected

    def test_bert_test_model(self, bert_test_model: PreTrainedModel) -> None:
        num_parameters = count_parameters(bert_test_model)
        assert num_parameters == 341_696

    def test_llama_test_model(self, llama_lm_test_model: PreTrainedModel) -> None:
        num_parameters = count_parameters(llama_lm_test_model)
        assert num_parameters == 598_336


class TestFormatNumber:
    @pytest.mark.parametrize(
        "number, expected",
        [
            (123, "123"),
            (999, "999"),
            (1_000, "1.0K"),
            (1_234, "1.2K"),
            (5_422, "5.4K"),
            # (999_999, "999.9K"),
            (1_000_000, "1.0M"),
            (1_234_567, "1.2M"),
            # (999_999_999, "999.9M"),
            (1_000_000_000, "1.0B"),
            (1_234_567_890, "1.2B"),
        ],
    )
    def test_mix_cases(self, number: int, expected: str) -> None:
        assert format_number(number) == expected
