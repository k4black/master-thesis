import pytest
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedModel, DataCollatorWithPadding
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
def bert_tiny_clf_model() -> PreTrainedModel:
    from transformers import BertForSequenceClassification, BertConfig
    try:
        config = BertConfig.from_pretrained(HF_BERT_TINY, num_labels=3, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(HF_BERT_TINY, config=config, local_files_only=True)
    except Exception:
        config = BertConfig.from_pretrained(HF_BERT_TINY, num_labels=3, local_files_only=False)
        model = BertForSequenceClassification.from_pretrained(HF_BERT_TINY, config=config, local_files_only=False)
    return model


@pytest.fixture
def bert_tiny_lm_model() -> PreTrainedModel:
    from transformers import BertForMaskedLM
    try:
        model = BertForMaskedLM.from_pretrained(HF_BERT_TINY, local_files_only=True)
    except Exception:
        model = BertForMaskedLM.from_pretrained(HF_BERT_TINY, local_files_only=False)
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
def random_dataloader(bert_tiny_tokenizer: PreTrainedTokenizer) -> DataLoader:
    return DataLoader(
        Dataset.from_list([
            {
                "input_ids": torch.randint(0, 100, (10,)),
                "attention_mask": torch.randint(0, 2, (10,)),
                "label": torch.randint(0, 3, ()),
            }
            for _ in range(8)
        ]),
        collate_fn=DataCollatorWithPadding(tokenizer=bert_tiny_tokenizer),
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
