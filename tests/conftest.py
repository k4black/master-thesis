import pytest
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel, DataCollatorWithPadding, PretrainedConfig
from datasets import Dataset


HF_TINY_BERT = "prajjwal1/bert-tiny"


@pytest.fixture
def bert_test_config() -> PretrainedConfig:
    from transformers import BertConfig
    return BertConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_size=64,
        intermediate_size=128,
        max_position_embeddings=128,
        vocab_size=4096,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=3,
    )


@pytest.fixture
def bert_test_tokenizer(bert_test_config: PretrainedConfig) -> PreTrainedTokenizer:
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(HF_TINY_BERT)
    # reduce vocab size to match the config
    tokenizer.vocab = {k: v for k, v in tokenizer.vocab.items() if v < bert_test_config.vocab_size}
    return tokenizer


@pytest.fixture()
def bert_test_model(bert_test_config: PretrainedConfig) -> PreTrainedModel:
    from transformers import BertModel
    model = BertModel(bert_test_config)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -10.0, 10.0, generator=generator)
    model.eval()
    return model


@pytest.fixture
def bert_lm_test_model(bert_test_config: PretrainedConfig) -> PreTrainedModel:
    from transformers import BertForMaskedLM
    model = BertForMaskedLM(bert_test_config)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -10.0, 10.0, generator=generator)
    model.eval()
    return model


@pytest.fixture
def bert_clf_test_model(bert_test_config: PretrainedConfig) -> PreTrainedModel:
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification(bert_test_config)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -10.0, 10.0, generator=generator)
    model.eval()
    return model


@pytest.fixture
def llama_test_config() -> PretrainedConfig:
    from transformers import LlamaConfig
    return LlamaConfig(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=64,
        intermediate_size=128,
        max_position_embeddings=128,
        vocab_size=4096,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )


@pytest.fixture
def llama_lm_test_model(llama_test_config: PretrainedConfig) -> PreTrainedModel:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM(llama_test_config)
    # init weights as random with a fixed seed
    generator = torch.Generator().manual_seed(42)
    for p in model.parameters():
        torch.nn.init.uniform_(p, -10.0, 10.0, generator=generator)
    model.eval()
    return model


@pytest.fixture
def simple_mnli_dataset() -> Dataset:
    from datasets import Dataset, Features, Value, ClassLabel
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
def random_input_batch() -> dict[str, torch.Tensor]:
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
def random_dataloader(bert_test_tokenizer: PreTrainedTokenizer) -> DataLoader:
    return DataLoader(
        Dataset.from_list([
            {
                "input_ids": torch.randint(0, 100, (10,)),
                "attention_mask": torch.randint(0, 2, (10,)),
                "label": torch.randint(0, 3, ()),
            }
            for _ in range(8)
        ]),
        collate_fn=DataCollatorWithPadding(tokenizer=bert_test_tokenizer),
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
