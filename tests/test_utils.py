import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from adaptive_pruning.utils import (
    count_nonzero_parameters,
    count_total_parameters,
    count_zero_parameters,
    format_number,
    measure_model_stats
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

    def test_bert_test_model(self, bert_test_model: PreTrainedModel) -> None:
        BERT_TEST_MODEL_SIZE = 341_696
        assert count_total_parameters(bert_test_model) == BERT_TEST_MODEL_SIZE
        assert count_zero_parameters(bert_test_model) == 0
        assert count_nonzero_parameters(bert_test_model) == BERT_TEST_MODEL_SIZE

        # nullify some weights
        bert_test_model.base_model.encoder.layer[0].attention.self.query.weight.data.fill_(0)
        assert count_total_parameters(bert_test_model) == BERT_TEST_MODEL_SIZE
        assert count_zero_parameters(bert_test_model) == 64 * 64
        assert count_nonzero_parameters(bert_test_model) == BERT_TEST_MODEL_SIZE - 64 * 64

    def test_llama_test_model(self, llama_lm_test_model: PreTrainedModel) -> None:
        LLAMA_TEST_MODEL_SIZE = 598_336
        assert count_total_parameters(llama_lm_test_model) == LLAMA_TEST_MODEL_SIZE
        assert count_zero_parameters(llama_lm_test_model) == 0
        assert count_nonzero_parameters(llama_lm_test_model) == LLAMA_TEST_MODEL_SIZE

        # nullify some weights
        llama_lm_test_model.base_model.layers[0].self_attn.q_proj.weight.data.fill_(0)
        assert count_total_parameters(llama_lm_test_model) == LLAMA_TEST_MODEL_SIZE
        assert count_zero_parameters(llama_lm_test_model) == 64 * 64
        assert count_nonzero_parameters(llama_lm_test_model) == LLAMA_TEST_MODEL_SIZE - 64 * 64


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
    def test_no_pruning_llama_test_model(self, llama_lm_test_model: PreTrainedModel) -> None:
        LLAMA_TEST_MODEL_SIZE = 598_336
        LLAMA_TEST_MODEL_BASE_SIZE = 336_192
        LLAMA_TEST_MODEL_LAYER_SIZE = 36_992
        assert (
            LLAMA_TEST_MODEL_SIZE
            > LLAMA_TEST_MODEL_BASE_SIZE
            > LLAMA_TEST_MODEL_LAYER_SIZE * len(llama_lm_test_model.base_model.layers)
        )

        # original_model_stats
        original_model_stats = measure_original_model_stats(llama_lm_test_model)
        assert original_model_stats["total"] == LLAMA_TEST_MODEL_SIZE
        assert original_model_stats["base"] == LLAMA_TEST_MODEL_BASE_SIZE
        for i, layer in enumerate(llama_lm_test_model.base_model.layers):
            assert original_model_stats[f"layer_{i}"] == LLAMA_TEST_MODEL_LAYER_SIZE

        # with original_model_stats provided
        sparsity_stats = measure_pruned_model_stats(llama_lm_test_model, original_model_stats)
        assert sparsity_stats["total"]["num_original_parameters"] == LLAMA_TEST_MODEL_SIZE
        assert sparsity_stats["total"]["num_parameters"] == LLAMA_TEST_MODEL_SIZE
        assert sparsity_stats["total"]["num_zero_parameters"] == 0
        assert sparsity_stats["total"]["num_nonzero_parameters"] == LLAMA_TEST_MODEL_SIZE
        assert sparsity_stats["base"]["num_original_parameters"] == LLAMA_TEST_MODEL_BASE_SIZE
        assert sparsity_stats["base"]["num_parameters"] == LLAMA_TEST_MODEL_BASE_SIZE
        assert sparsity_stats["base"]["num_zero_parameters"] == 0
        assert sparsity_stats["base"]["num_nonzero_parameters"] == LLAMA_TEST_MODEL_BASE_SIZE
        for i, layer in enumerate(llama_lm_test_model.base_model.layers):
            assert sparsity_stats[f"layer_{i}"]["num_original_parameters"] == LLAMA_TEST_MODEL_LAYER_SIZE
            assert sparsity_stats[f"layer_{i}"]["percentage_original_pruned"] == 0.0
            assert sparsity_stats[f"layer_{i}"]["num_parameters"] == LLAMA_TEST_MODEL_LAYER_SIZE
            assert sparsity_stats[f"layer_{i}"]["num_zero_parameters"] == 0
            assert sparsity_stats[f"layer_{i}"]["num_nonzero_parameters"] == LLAMA_TEST_MODEL_LAYER_SIZE

        # without original_model_stats provided
        sparsity_stats = measure_pruned_model_stats(llama_lm_test_model)
        assert sparsity_stats["total"]["num_original_parameters"] is None
        assert sparsity_stats["total"]["num_parameters"] == LLAMA_TEST_MODEL_SIZE

    def test_pruning_llama_test_model(self, llama_lm_test_model: PreTrainedModel) -> None:
        LLAMA_TEST_MODEL_SIZE = 598_336
        LLAMA_TEST_MODEL_BASE_SIZE = 336_192
        LLAMA_TEST_MODEL_LAYER_SIZE = 36_992
        assert (
            LLAMA_TEST_MODEL_SIZE
            > LLAMA_TEST_MODEL_BASE_SIZE
            > LLAMA_TEST_MODEL_LAYER_SIZE * len(llama_lm_test_model.base_model.layers)
        )

        # original_model_stats
        original_model_stats = measure_original_model_stats(llama_lm_test_model)
        assert original_model_stats["total"] == LLAMA_TEST_MODEL_SIZE
        assert original_model_stats["base"] == LLAMA_TEST_MODEL_BASE_SIZE
        assert original_model_stats["layer_0"] == LLAMA_TEST_MODEL_LAYER_SIZE

        # nullify some weights
        llama_lm_test_model.base_model.layers[0].self_attn.q_proj.weight.data.fill_(0)
        llama_lm_test_model.base_model.layers[1].self_attn.q_proj.weight.data.fill_(0)
        llama_lm_test_model.base_model.layers[1].self_attn.o_proj.weight.data = torch.empty(0, 0)

        # with original_model_stats provided
        sparsity_stats = measure_pruned_model_stats(llama_lm_test_model, original_model_stats)
        assert sparsity_stats["total"]["num_original_parameters"] == LLAMA_TEST_MODEL_SIZE
        assert sparsity_stats["total"]["num_parameters"] == LLAMA_TEST_MODEL_SIZE - 64 * 64
        assert sparsity_stats["total"]["num_zero_parameters"] == 64 * 64 * 2
        assert sparsity_stats["total"]["percentage_original_pruned"] == 64 * 64 / LLAMA_TEST_MODEL_SIZE * 100
        assert sparsity_stats["total"]["percentage_zero"] == 64 * 64 * 2 / (LLAMA_TEST_MODEL_SIZE - 64 * 64) * 100

        assert sparsity_stats["layer_0"]["num_original_parameters"] == LLAMA_TEST_MODEL_LAYER_SIZE
        assert sparsity_stats["layer_0"]["num_parameters"] == LLAMA_TEST_MODEL_LAYER_SIZE
        assert sparsity_stats["layer_0"]["num_zero_parameters"] == 64 * 64
        assert sparsity_stats["layer_0"]["percentage_original_pruned"] == 0.0
        assert sparsity_stats["layer_0"]["percentage_zero"] == 64 * 64 / LLAMA_TEST_MODEL_LAYER_SIZE * 100

        assert sparsity_stats["layer_1"]["num_original_parameters"] == LLAMA_TEST_MODEL_LAYER_SIZE
        assert sparsity_stats["layer_1"]["num_parameters"] == LLAMA_TEST_MODEL_LAYER_SIZE - 64 * 64
        assert sparsity_stats["layer_1"]["num_zero_parameters"] == 64 * 64
        assert sparsity_stats["layer_1"]["percentage_original_pruned"] == 64 * 64 / LLAMA_TEST_MODEL_LAYER_SIZE * 100
        assert sparsity_stats["layer_1"]["percentage_zero"] == 64 * 64 / (LLAMA_TEST_MODEL_LAYER_SIZE - 64 * 64) * 100

    # def test_print_measure_table(self, llama_lm_test_model: PreTrainedModel, capsys: pytest.CaptureFixture) -> None:
    #     # original_model_stats
    #     original_model_stats = measure_original_model_stats(llama_lm_test_model)
    #     # nullify some weights
    #     llama_lm_test_model.base_model.layers[0].self_attn.q_proj.weight.data.fill_(0)
    #     llama_lm_test_model.base_model.layers[1].self_attn.q_proj.weight.data.fill_(0)
    #     llama_lm_test_model.base_model.layers[1].self_attn.o_proj.weight.data = torch.empty(0, 0)
    #     # with original_model_stats provided
    #     sparsity_stats = measure_pruned_model_stats(llama_lm_test_model, original_model_stats)
    #     print_measure_table(sparsity_stats)
    #     captured = capsys.readouterr()
    #     assert 'Module' in captured.out
    #     assert '%Pruned' in captured.out
    #     # ...

    #     sparsity_stats = measure_pruned_model_stats(llama_lm_test_model, None)
    #     print_measure_table(sparsity_stats)
    #     captured = capsys.readouterr()
    #     assert 'Module' in captured.out
    #     assert '%Pruned' in captured.out
    #     # ...
