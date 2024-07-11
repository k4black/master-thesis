import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, PretrainedConfig, PreTrainedModel, PreTrainedTokenizer


HF_TINY_BERT = "prajjwal1/bert-tiny"
HF_TINY_LLAMA = "TinyLlama/TinyLlama_v1.1"

TEST_MODELS_NUM_LAYERS = 2
TEST_MODELS_NUM_ATTENTION_HEADS = 4
TEST_MODELS_HIDDEN_SIZE = 64
TEST_MODELS_INTERMEDIATE_SIZE = 128
TEST_MODELS_VOCAB_SIZE = 4096


@pytest.fixture
def test_config_bert() -> PretrainedConfig:
    from transformers import BertConfig

    return BertConfig(
        num_hidden_layers=TEST_MODELS_NUM_LAYERS,
        num_attention_heads=TEST_MODELS_NUM_ATTENTION_HEADS,
        hidden_size=TEST_MODELS_HIDDEN_SIZE,
        intermediate_size=TEST_MODELS_INTERMEDIATE_SIZE,
        max_position_embeddings=128,
        vocab_size=TEST_MODELS_VOCAB_SIZE,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=3,
    )


@pytest.fixture
def test_tokenizer_bert(test_config_bert: PretrainedConfig) -> PreTrainedTokenizer:
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(HF_TINY_BERT)
    # reduce vocab size to match the config
    tokenizer.vocab = {k: v for k, v in tokenizer.vocab.items() if v < test_config_bert.vocab_size}
    return tokenizer


@pytest.fixture()
def test_base_model_bert(test_config_bert: PretrainedConfig) -> PreTrainedModel:
    from transformers import BertModel

    model = BertModel(test_config_bert)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -1.0, 1.0, generator=generator)
    model.eval()
    return model


@pytest.fixture
def test_lm_model_bert(test_config_bert: PretrainedConfig) -> PreTrainedModel:
    from transformers import BertForMaskedLM

    model = BertForMaskedLM(test_config_bert)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -1.0, 1.0, generator=generator)
    model.eval()
    return model


@pytest.fixture
def test_clf_model_bert(test_config_bert: PretrainedConfig) -> PreTrainedModel:
    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification(test_config_bert)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -1.0, 1.0, generator=generator)
    model.eval()
    return model


@pytest.fixture
def test_config_llama() -> PretrainedConfig:
    from transformers import LlamaConfig

    return LlamaConfig(
        num_hidden_layers=TEST_MODELS_NUM_LAYERS,
        num_attention_heads=TEST_MODELS_NUM_ATTENTION_HEADS,
        num_key_value_heads=2,
        hidden_size=TEST_MODELS_HIDDEN_SIZE,
        intermediate_size=TEST_MODELS_INTERMEDIATE_SIZE,
        max_position_embeddings=128,
        vocab_size=TEST_MODELS_VOCAB_SIZE,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )


@pytest.fixture
def test_tokenizer_llama(test_config_llama: PretrainedConfig) -> PreTrainedTokenizer:
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained(HF_TINY_LLAMA, legacy=True)

    return tokenizer


@pytest.fixture
def test_base_model_llama(test_config_llama: PretrainedConfig) -> PreTrainedModel:
    from transformers import LlamaModel

    model = LlamaModel(test_config_llama)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -1.0, 1.0, generator=generator)
    model.eval()
    return model


@pytest.fixture
def test_lm_model_llama(test_config_llama: PretrainedConfig) -> PreTrainedModel:
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM(test_config_llama)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -1.0, 1.0, generator=generator)
    model.eval()
    return model


@pytest.fixture(params=["bert", "llama"])
def test_base_model(
    request: pytest.FixtureRequest, test_base_model_bert: PreTrainedModel, test_base_model_llama: PreTrainedModel
) -> PreTrainedModel:
    if request.param == "bert":
        return test_base_model_bert
    elif request.param == "llama":
        return test_base_model_llama
    else:
        raise ValueError(f"Unknown model type: {request.param}")


@pytest.fixture(params=["bert", "llama"])
def test_lm_model(
    request: pytest.FixtureRequest, test_lm_model_bert: PreTrainedModel, test_lm_model_llama: PreTrainedModel
) -> PreTrainedModel:
    if request.param == "bert":
        return test_lm_model_bert
    elif request.param == "llama":
        return test_lm_model_llama
    else:
        raise ValueError(f"Unknown model type: {request.param}")


@pytest.fixture
def simple_mnli_dataset() -> Dataset:
    from datasets import ClassLabel, Dataset, Features, Value

    return Dataset.from_dict(
        {
            "premise": ["I like turtles", "I like pizza", "I like bananas", "I am a human"],
            "hypothesis": ["Turtles are cool", "Pizza is tasty", "Bananas are yellow", "I am a robot"],
            "label": [0, 1, 2, 0],
        },
        features=Features(
            {
                "premise": Value("string"),
                "hypothesis": Value("string"),
                "label": ClassLabel(names=["entailment", "neutral", "contradiction"]),
            }
        ),
    )


@pytest.fixture
def random_batch() -> dict[str, torch.Tensor]:
    """
    Returns a dictionary with random input_ids, attention_mask and labels tensors.
    :return: [4, 10] input_ids, [4, 10] attention_mask, [4, 10] labels
    """
    import torch

    return {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.randint(0, 2, (4, 10)),
    }


@pytest.fixture
def random_clf_batch() -> dict[str, torch.Tensor]:
    """
    Returns a dictionary with random input_ids, attention_mask and label tensors.
    :return: [4, 10] input_ids, [4, 10] attention_mask, [4] label
    """
    import torch

    return {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.randint(0, 2, (4, 10)),
        "label": torch.randint(0, 3, (4,)),
    }


@pytest.fixture
def random_lm_batch() -> dict[str, torch.Tensor]:
    """
    Returns a dictionary with random input_ids, attention_mask and labels tensors.
    :return: [4, 10] input_ids, [4, 10] attention_mask, [4, 10] labels
    """
    import torch

    return {
        "input_ids": torch.randint(0, 100, (4, 10)),
        "attention_mask": torch.randint(0, 2, (4, 10)),
        "labels": torch.randint(0, 100, (4, 10)),
    }


@pytest.fixture
def random_clf_dataloader(test_tokenizer_bert: PreTrainedTokenizer) -> DataLoader:
    return DataLoader(
        Dataset.from_list(
            [
                {
                    "input_ids": torch.randint(1, 100, (10,)),
                    "attention_mask": torch.randint(1, 2, (10,)),
                    "label": torch.randint(0, 3, ()),
                }
                for _ in range(8)
            ]
        ),
        collate_fn=DataCollatorWithPadding(tokenizer=test_tokenizer_bert),
        batch_size=4,
    )


@pytest.fixture
def random_lm_dataloader(test_tokenizer_bert: PreTrainedTokenizer) -> DataLoader:
    return DataLoader(
        Dataset.from_list(
            [
                {
                    "input_ids": torch.randint(1, 100, (10,)),
                    "attention_mask": torch.randint(1, 2, (10,)),
                    "labels": torch.randint(1, 100, (10,)),
                }
                for _ in range(8)
            ]
        ),
        collate_fn=DataCollatorWithPadding(tokenizer=test_tokenizer_bert),
        batch_size=4,
    )


# Pytest configuration


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-slow",
        action="store_true",
        dest="skip_slow",
        default=False,
        help="skip slow tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    skip_slow = pytest.mark.skip(reason="--skip-slow option was provided")
    for item in items:
        if "slow" in item.keywords and config.getoption("--skip-slow"):
            item.add_marker(skip_slow)
