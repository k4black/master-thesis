from collections import defaultdict
from typing import Callable

import neptune
from datasets import Dataset
from neptune.types import File
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from adaptive_pruning.importance import (
    ComponentsImportance,
    ComponentsInfo,
    ComponentsToPrune,
    collect_mask_gradients,
    get_insert_pruning_masks,
)
from adaptive_pruning.nullify import nullify_hidden_states, nullify_attention_heads, nullify_ffn_neurons, \
    nullify_attention_layers, nullify_ffn_layers
from adaptive_pruning.utils import tensor_to_list, count_zero_parameters


class PruningTrainerCallback(TrainerCallback):
    """
    Callback for pruning models during training.
    First it insert temporary hooks to collect the gradients/activations in the model and prune as-we-go.
    On each step
      1. it checks if it is time to prune the model (make certain number of iterations of pruning)
      2. Collect the gradients (or use collected during the training)
      3. Prune the model
    """

    def __init__(
        self,
        target_ratio: float,
        components: list[str],  # ["attn_heads", "attn_layers", "ffn_layers", "ffn_neurons", "hidden_states"]
        strategy: str,  # ["mask_grads", "full_grads", "weights", "activations", "random"]
        average: str,  # ["fisher_info", "sum", "mean", "max", "entropy"]
        overlap: str,  # ["fixed", "relative", "meta", "trainable"]
        dataset: Dataset,
        data_collator: Callable,
        num_samples: int,
        batch_size: int,
        is_uniform: bool = False,
        round_to: int = 1,  # at the very end, the very last pruning round only
        num_epochs: float | None = None,  # have to be < than num_train_epochs, default to
        num_iterations: int = 1,  # number of pruning steps across the pruning_num_epochs
        prune_before_training: bool = False,
        nullify_model_weights: bool = True,
        *,
        neptune_run: neptune.Run = None,
    ):
        self.target_ratio = target_ratio
        self.components = components
        self.strategy = strategy
        assert self.strategy in ["grads", "mask_grads"], "Only 'mask_grads' strategies is supported"
        self.average = average
        self.overlap = overlap
        self.num_epochs = num_epochs
        self.round_to = round_to
        self.num_iterations = num_iterations
        self.neptune_run = neptune_run
        self.is_uniform = is_uniform
        self.prune_before_training = prune_before_training
        self.nullify_model_weights = nullify_model_weights

        self.dataset = dataset
        self.data_collator = data_collator
        self.num_samples = num_samples
        self.batch_size = batch_size

        # to be filled during the on_init_end
        self._current_pruned_ratio = 0.0
        self._max_pruning_step = None
        self._fake_pruning_masks = {}
        self._pruned_components: ComponentsToPrune | None = None
        self._fake_pruning_hooks = []
        self._pruning_step_to_pruning_ratio = {}
        self._model = None

    @property
    def pruning_components(self) -> ComponentsToPrune | None:
        return self._pruned_components

    def _masks_require_grad(self, require_grad: bool) -> None:
        for mask in self._fake_pruning_masks.values():
            mask.requires_grad_(require_grad)

    def _init_pruning_params(
        self,
        args: TrainingArguments,
        state: TrainerState,
        model: PreTrainedModel,
    ) -> None:
        """
        We need to initialize the pruning parameters after the trainer is initialized,
        however the "on_init_end" sometime is called before the trainer is fully initialized.
        """
        # assert state.max_steps, "The trainer has to be initialized before the pruning callback"
        self._model = model

        # calculate the percent to prune
        if self.num_epochs is None:
            self.num_epochs = args.num_train_epochs

        if self.num_iterations == 1:
            # only one pruning step - prune than train
            self._pruning_step_to_pruning_ratio = {0: self.target_ratio}
            self._max_pruning_step = 0
        else:
            if args.num_train_epochs or self.prune_before_training:
                share_to_prune = self.target_ratio / self.num_iterations
                self._pruning_step_to_pruning_ratio = {i: (i + 1) * share_to_prune for i in range(self.num_iterations)}
                self._max_pruning_step = 0
            else:
                # multiple pruning steps - prune gradually, same percent for each step
                percent_of_pruning_epochs = self.num_epochs / args.num_train_epochs
                max_pruning_step = int(state.max_steps * percent_of_pruning_epochs)
                # divide the pruning into equal steps
                if self.num_iterations > max_pruning_step:
                    self.num_iterations = max_pruning_step
                steps_between_pruning = max_pruning_step // self.num_iterations
                share_to_prune = self.target_ratio / self.num_iterations
                self._pruning_step_to_pruning_ratio = {
                    (i + 1) * steps_between_pruning: (i + 1) * share_to_prune for i in range(self.num_iterations)
                }
                self._max_pruning_step = self.num_iterations * steps_between_pruning

        # create masks for the model and insert the hooks
        self._fake_pruning_masks, self._fake_pruning_hooks = get_insert_pruning_masks(self._model, require_grads=False)

        # init _pruned_components
        self._pruned_components = ComponentsToPrune(
            attention_heads_to_prune=defaultdict(list),
            attention_layers_to_prune=[],
            ffn_neurons_to_prune=defaultdict(list),
            ffn_layers_to_prune=[],
            hidden_states_to_prune=[],
        )
        print(f"PRUNING: initialized with {self.num_iterations} iterations, ")
        print(f"PRUNING: max pruning step {self._max_pruning_step}, ")
        print(f"PRUNING: pruning schedule {self._pruning_step_to_pruning_ratio}")

    def _fake_prune_model(self, to_prune_target: float, round_to: int, step_id: int = 0, optimizer=None) -> None:
        """
        Zero-out masks in the model to simulate pruning.
        """

        data_loader = DataLoader(
            self.dataset.shuffle(seed=step_id).select(range(self.num_samples)),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

        # collect the gradients
        # TODO: temp skip info collection
        components_info: ComponentsInfo = collect_mask_gradients(
            model=self._model,
            dataloader=data_loader,
        )
        if optimizer:
            optimizer.zero_grad()
        # get importance of the components and select the components to prune
        components_importance = ComponentsImportance.from_info(components_info, how_to_average=self.average)

        components_to_prune = ComponentsToPrune.from_importance(
            components_importance=components_importance,
            pruning_ratio_target=to_prune_target,
            pruning_components=self.components,
            round_to=round_to,
            is_uniform=self.is_uniform,
            how_to_overlap=self.overlap,
            config=self._model.config,
            already_pruned_components=self._pruned_components,
        )

        # prune the model (fake, just nullify the mask)
        if "attn_heads" in self.components:
            for layer, heads in components_to_prune.attention_heads_to_prune.items():
                self._fake_pruning_masks["attn_heads"][layer, heads] = 0.0
        if "attn_layers" in self.components:
            layers = components_to_prune.attention_layers_to_prune
            self._fake_pruning_masks["attn_layers"][layers] = 0.0
        if "ffn_neurons" in self.components:
            for layer, neurons in components_to_prune.ffn_neurons_to_prune.items():
                self._fake_pruning_masks["ffn_neurons"][layer, neurons] = 0.0
        if "ffn_layers" in self.components:
            layers = components_to_prune.ffn_layers_to_prune
            self._fake_pruning_masks["ffn_layers"][layers] = 0.0
        if "hidden_states" in self.components:
            states = components_to_prune.hidden_states_to_prune
            self._fake_pruning_masks["hidden_states"][states] = 0.0

        # update _pruned_components, extend the pruned components
        for layer, heads in components_to_prune.attention_heads_to_prune.items():
            self._pruned_components.attention_heads_to_prune[layer].extend(heads)
        self._pruned_components.attention_layers_to_prune.extend(components_to_prune.attention_layers_to_prune)
        for layer, neurons in components_to_prune.ffn_neurons_to_prune.items():
            self._pruned_components.ffn_neurons_to_prune[layer].extend(neurons)
        self._pruned_components.ffn_layers_to_prune.extend(components_to_prune.ffn_layers_to_prune)
        self._pruned_components.hidden_states_to_prune.extend(components_to_prune.hidden_states_to_prune)

        # nullify the weights in the model itself to optimize the performance
        if self.nullify_model_weights:
            current_num_zero_weights = count_zero_parameters(self._model, require_grad=None)
            print(f"PRUNING: nullifying the weights in the model ({current_num_zero_weights} zero weights)")
            if 'attn_heads' in self.components:
                nullify_attention_heads(self._model, self._pruned_components.attention_heads_to_prune)
            if 'attn_layers' in self.components:
                nullify_attention_layers(self._model, self._pruned_components.attention_layers_to_prune)
            if 'ffn_neurons' in self.components:
                nullify_ffn_neurons(self._model, self._pruned_components.ffn_neurons_to_prune)
            if 'ffn_layers' in self.components:
                nullify_ffn_layers(self._model, self._pruned_components.ffn_layers_to_prune)
            if 'hidden_states' in self.components:
                nullify_hidden_states(self._model, self._pruned_components.hidden_states_to_prune)
            after_num_zero_weights = count_zero_parameters(self._model, require_grad=None)
            print(f"PRUNING: nullified the weights in the model ({after_num_zero_weights} zero weights)")

        # log the pruning
        self._current_pruned_ratio = to_prune_target  # TODO: account for actual pruning rate (round_to)

        if self.neptune_run:
            for name, value in components_info._asdict().items():
                self.neptune_run[f"pruning/step_{step_id}_{name}"].upload(File.as_pickle(tensor_to_list(value)))
            for name, value in components_importance._asdict().items():
                self.neptune_run[f"pruning/step_{step_id}_{name}"].upload(File.as_pickle(tensor_to_list(value)))
            for name, value in components_to_prune._asdict().items():
                self.neptune_run[f"pruning/step_{step_id}_{name}"].upload(File.as_pickle(value))

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        """See _init_pruning_params"""
        pass

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        assert "model" in kwargs, "The model and tokenizer have to be passed to the callback"
        self._init_pruning_params(args, state, kwargs["model"])
        # TODO: add optimizer? to zero the gradients

        # if have no epochs/do_train=False but iterations >= 1, prune the model at the beginning in multiple steps
        if state.num_train_epochs == 0 or self.prune_before_training:
            for step, target_pruning_ratio in self._pruning_step_to_pruning_ratio.items():
                to_prune_now = target_pruning_ratio - self._current_pruned_ratio
                # verbose the pruning
                print(
                    f"PRUNING: step 0*{step}/{state.max_steps} (max {self._max_pruning_step}), "
                    f"pruning now {to_prune_now:.1%} (to have {target_pruning_ratio:.1%}), round to {self.round_to}"
                )
                # prune the model
                self._fake_prune_model(target_pruning_ratio, self.round_to, step_id=step)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        # check if it is time to prune (0th step is pruned in on_train_begin)
        if state.global_step in self._pruning_step_to_pruning_ratio and not self.prune_before_training:
            # get the percent to prune this round
            target_pruning_ratio = self._pruning_step_to_pruning_ratio[state.global_step]
            to_prune_now = target_pruning_ratio - self._current_pruned_ratio

            # verbose the pruning
            print(
                f"PRUNING: step {state.global_step}/{state.max_steps} (max {self._max_pruning_step}), "
                f"pruning now {to_prune_now:.1%} (to have {target_pruning_ratio:.1%}), round to {self.round_to}"
            )

            # prune the model
            self._fake_prune_model(target_pruning_ratio, self.round_to, step_id=state.global_step)

        self._model.zero_grad()

    def on_optimizer_step(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        # TODO: collect the gradients from the prev step
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        # print the final pruning ratio
        print(f"PRUNING: final pruning ratio {self._current_pruned_ratio:.1%} in {self.num_iterations} iterations")
        # remove the hooks
        for hook in self._fake_pruning_hooks:
            hook.remove()
        for mask in self._fake_pruning_masks.values():
            mask.detach_()

        self._fake_pruning_hooks = []
        self._fake_pruning_masks.clear()

        # update saved pruned components to match round_to

    def __del__(self) -> None:
        # remove the hooks
        if hasattr(self, "_fake_pruning_hooks"):
            for hook in self._fake_pruning_hooks:
                hook.remove()
        if hasattr(self, "_fake_pruning_masks"):
            for mask in self._fake_pruning_masks.values():
                mask.detach_()
