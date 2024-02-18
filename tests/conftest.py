import pytest
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import Dataset


# Fixtures


HF_BERT_TINY = "prajjwal1/bert-tiny"


@pytest.fixture
def bert_tiny_tokenizer() -> PreTrainedTokenizer:
    from transformers import BertTokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained(HF_BERT_TINY, local_files_only=True)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(HF_BERT_TINY, local_files_only=False)
    return tokenizer


@pytest.fixture
def bert_tiny_model() -> PreTrainedModel:
    from transformers import BertModel
    try:
        model = BertModel.from_pretrained(HF_BERT_TINY, local_files_only=True)
    except Exception:
        model = BertModel.from_pretrained(HF_BERT_TINY, local_files_only=False)
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
def random_input() -> dict[str, torch.Tensor]:
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
