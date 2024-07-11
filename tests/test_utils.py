import pytest
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from adaptive_pruning.utils import (
    count_nonzero_parameters,
    count_total_parameters,
    count_zero_parameters,
    format_number,
    measure_model_stats,
)


class TestDifferentCountParameters:

    @pytest.mark.parametrize(
        "module, expected",
        [
            (nn.Linear(10, 5, bias=False), 10 * 5),
            (nn.Linear(10, 5, bias=True), 10 * 5 + 5),
            (nn.Sequential(nn.Linear(10, 5, bias=False), nn.Linear(5, 4, bias=False)), 10 * 5 + 5 * 4),
            (nn.MultiheadAttention(10, 5, bias=False), 10 * 5 * 4 * 2),
        ],
    )
    def test_pytorch_module(self, module: nn.Module, expected: int) -> None:
        for p in module.parameters():
            nn.init.constant_(p, 1)
        assert count_total_parameters(module) == expected
        assert count_zero_parameters(module) == 0
        assert count_nonzero_parameters(module) == expected

        for p in module.parameters():
            nn.init.constant_(p, 0)
        assert count_total_parameters(module) == expected
        assert count_zero_parameters(module) == expected
        assert count_nonzero_parameters(module) == 0

    def test_bert_test_model(self, test_base_model_bert: PreTrainedModel) -> None:
        BERT_TEST_MODEL_SIZE = 341_696
        assert count_total_parameters(test_base_model_bert) == BERT_TEST_MODEL_SIZE
        assert count_zero_parameters(test_base_model_bert) == 0
        assert count_nonzero_parameters(test_base_model_bert) == BERT_TEST_MODEL_SIZE

        # nullify some weights
        test_base_model_bert.base_model.encoder.layer[0].attention.self.query.weight.data.fill_(0)
        assert count_total_parameters(test_base_model_bert) == BERT_TEST_MODEL_SIZE
        assert count_zero_parameters(test_base_model_bert) == 64 * 64
        assert count_nonzero_parameters(test_base_model_bert) == BERT_TEST_MODEL_SIZE - 64 * 64

    def test_llama_test_model(self, test_lm_model_llama: PreTrainedModel) -> None:
        LLAMA_TEST_MODEL_SIZE = 598_336
        assert count_total_parameters(test_lm_model_llama) == LLAMA_TEST_MODEL_SIZE
        assert count_zero_parameters(test_lm_model_llama) == 0
        assert count_nonzero_parameters(test_lm_model_llama) == LLAMA_TEST_MODEL_SIZE

        # nullify some weights
        test_lm_model_llama.base_model.layers[0].self_attn.q_proj.weight.data.fill_(0)
        assert count_total_parameters(test_lm_model_llama) == LLAMA_TEST_MODEL_SIZE
        assert count_zero_parameters(test_lm_model_llama) == 64 * 64
        assert count_nonzero_parameters(test_lm_model_llama) == LLAMA_TEST_MODEL_SIZE - 64 * 64


class TestFormatNumber:
    @pytest.mark.parametrize(
        "number, expected",
        [
            (123, "123"),
            (999, "999"),
            (1_000, "1.0K"),
            (1_100, "1.1K"),
            (1_234, "1.2K"),
            (5_422, "5.4K"),
            # (999_999, "999.9K"),
            (1_000_000, "1.0M"),
            (1_089_000, "1.1M"),
            (1_234_567, "1.2M"),
            # (999_999_999, "999.9M"),
            (1_000_000_000, "1.0B"),
            (1_090_000_000, "1.1B"),
            (1_234_567_890, "1.2B"),
        ],
    )
    def test_mix_cases(self, number: int, expected: str) -> None:
        assert format_number(number) == expected


class TestMeasureModels:
    LLAMA_TEST_MODEL_SIZE = 598_336
    LLAMA_TEST_MODEL_BASE_SIZE = 336_192
    LLAMA_TEST_MODEL_LAYER_SIZE = 36_992

    def test_measure_model_stats_original(
        self, test_lm_model_llama: PreTrainedModel, test_tokenizer_llama: PreTrainedTokenizer
    ):
        stats, total_stats = measure_model_stats(test_lm_model_llama, test_tokenizer_llama, print_results=True)
        assert "total" in stats
        assert total_stats["params"] == self.LLAMA_TEST_MODEL_SIZE

    def test_measure_model_stats_with_pruning(
        self, test_lm_model_llama: PreTrainedModel, test_tokenizer_llama: PreTrainedTokenizer
    ):
        hidden_size = test_lm_model_llama.config.hidden_size
        test_lm_model_llama.base_model.layers[1].mlp.up_proj.weight.data = torch.empty(10, hidden_size)
        test_lm_model_llama.base_model.layers[1].mlp.gate_proj.weight.data = torch.empty(10, hidden_size)
        test_lm_model_llama.base_model.layers[1].mlp.down_proj.weight.data = torch.empty(hidden_size, 10)
        stats, total_stats = measure_model_stats(test_lm_model_llama, test_tokenizer_llama, print_results=True)
        assert stats[1]["ffn_n_params"] == hidden_size * 10 * 3
        assert total_stats["params"] < self.LLAMA_TEST_MODEL_SIZE

    def test_measure_model_stats_with_zeroing(
        self, test_lm_model_llama: PreTrainedModel, test_tokenizer_llama: PreTrainedTokenizer
    ):
        hidden_size = test_lm_model_llama.config.hidden_size
        test_lm_model_llama.base_model.layers[0].self_attn.q_proj.weight.data.fill_(0)
        stats, total_stats = measure_model_stats(test_lm_model_llama, test_tokenizer_llama, print_results=True)
        assert stats[0]["attn_heads_n_zero_params"] == hidden_size * hidden_size
        assert total_stats["zero_params"] >= hidden_size * hidden_size

    def test_measure_model_stats_comparison(
        self, test_lm_model_llama: PreTrainedModel, test_tokenizer_llama: PreTrainedTokenizer
    ):
        hidden_size = test_lm_model_llama.config.hidden_size
        original_stats, _ = measure_model_stats(test_lm_model_llama, test_tokenizer_llama, print_results=False)
        test_lm_model_llama.base_model.layers[1].mlp.up_proj.weight.data = torch.empty(10, hidden_size)
        test_lm_model_llama.base_model.layers[1].mlp.gate_proj.weight.data = torch.empty(10, hidden_size)
        test_lm_model_llama.base_model.layers[1].mlp.down_proj.weight.data = torch.empty(hidden_size, 10)
        test_lm_model_llama.base_model.layers[0].self_attn.q_proj.weight.data.fill_(0)
        pruned_stats, pruned_total_stats = measure_model_stats(
            test_lm_model_llama, test_tokenizer_llama, original_model_stats=original_stats, print_results=True
        )
        assert pruned_stats[1]["ffn_n_params"] == hidden_size * 10 * 3
        assert pruned_total_stats["params"] < self.LLAMA_TEST_MODEL_SIZE
