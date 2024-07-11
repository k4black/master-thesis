from dataclasses import dataclass, field
from typing import Any, Literal

import datasets
from torch.utils.data import Dataset as TorchDataset, IterableDataset as TorchIterableDataset
from transformers import Trainer, TrainingArguments


@dataclass
class PruningTrainingArguments(TrainingArguments):
    pruning_ratio: float = field(default=0.0)
    pruning_components: Literal["attn-heads", "attn-layers", "ffn-layers", "ffn-neurons", "hidden-states"] = field(
        default="ffn-neurons"
    )
    pruning_round_to: int = field(default=1)
    pruning_strategy: Literal["mask-grads", "full-grads", "weights", "activations", "random"] = field(default="masks")
    pruning_average: Literal["fisher-info", "sum", "mean", "max", "entropy"] = field(default="fisher-info")
    pruning_overlap: Literal["fixed", "relative", "meta"] = field(default="fixed")
    # have to be < than num_train_epochs
    pruning_num_epochs: int = field(default=1)
    # number of pruning steps across the pruning_num_epochs,
    #   if 1, then pruning is done only once before the training starts
    pruning_num_iterations: int = field(default=1)


class PruningTrainer(Trainer):
    def __init__(
        self,
        *,  # prevent positional arguments
        args: PruningTrainingArguments,
        # if pruning_dataset is None, then the training dataset is used for pruning (from prev iteration)
        pruning_dataset: TorchDataset | TorchIterableDataset | datasets.Dataset | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(args=args, **kwargs)
        self.pruning_dataset = pruning_dataset
