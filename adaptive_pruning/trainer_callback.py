from collections import defaultdict

import neptune
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
from adaptive_pruning.pruning import (
    prune_attention_heads,
    prune_attention_layers,
    prune_ffn_layers,
    prune_ffn_neurons,
    prune_hidden_states,
)
from adaptive_pruning.utils import tensor_to_list


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
        strategy: str,  # ["mask-grads", "full-grads", "weights", "activations", "random"]
        average: str,  # ["fisher-info", "sum", "mean", "max", "entropy"]
        overlap: str,  # ["fixed", "relative", "meta", "trainable"]
        data_loader: DataLoader,
        is_uniform: bool = False,
        round_to: int = 1,  # at the very end, the very last pruning round only
        num_epochs: float | None = None,  # have to be < than num_train_epochs, default to
        num_iterations: int = 1,  # number of pruning steps across the pruning_num_epochs
        *,
        neptune_run: neptune.Run = None,
    ):
        self.target_ratio = target_ratio
        self.components = components
        self.strategy = strategy
        self.average = average
        self.overlap = overlap
        self.num_epochs = num_epochs
        self.round_to = round_to
        self.num_iterations = num_iterations
        self.neptune_run = neptune_run
        self.is_uniform = is_uniform

        self.data_loader = data_loader

        # to be filled during the on_init_end
        self._current_pruned_ratio = 0.0
        self._max_pruning_step = None
        self._fake_pruning_masks = {}
        self._pruned_components: ComponentsToPrune = None
        self._fake_pruning_hooks = []
        self._pruning_step_to_share = {}
        self._model = None

    @property
    def pruning_masks(self) -> dict:
        return self._fake_pruning_masks

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
            self._pruning_step_to_share = {0: self.target_ratio}
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
            self._pruning_step_to_share = {
                i * steps_between_pruning: share_to_prune for i in range(self.num_iterations)
            }
            self._max_pruning_step = (self.num_iterations - 1) * steps_between_pruning

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
        print(f"PRUNING: pruning schedule {self._pruning_step_to_share}")

    def _fake_prune_model(self, to_prune: float, round_to: int, step_id: int = 0) -> None:
        """
        Zero-out masks in the model to simulate pruning.
        """
        # collect the gradients
        components_info: ComponentsInfo = collect_mask_gradients(
            model=self._model,
            dataloader=self.data_loader,
            pruning_masks_hooks=None,  # create new masks each time to avoid the error with _fake_pruning_masks
            remove_hooks=False,
        )
        # get importance of the components and select the components to prune
        components_importance = ComponentsImportance.from_info(components_info, how_to_average=self.average)

        # Select the components to prune, first skip pruned components by set Importances to Inf
        # TODO: fix the skip_pruned_components
        components_importance.attention_layers_importance[self._pruned_components.attention_layers_to_prune] = float(
            "inf"
        )
        for layer, heads in self._pruned_components.attention_heads_to_prune.items():
            components_importance.attention_heads_importance[layer, heads] = float("inf")
        components_importance.ffn_layers_importance[self._pruned_components.ffn_layers_to_prune] = float("inf")
        for layer, neurons in self._pruned_components.ffn_neurons_to_prune.items():
            components_importance.ffn_neurons_importance[layer, neurons] = float("inf")
        components_importance.hidden_states_importance[self._pruned_components.hidden_states_to_prune] = float("inf")

        components_to_prune = ComponentsToPrune.from_importance(
            components_importance=components_importance,
            pruning_ratio=self.target_ratio,
            pruning_components=self.components,
            round_to=round_to,
            is_uniform=self.is_uniform,
            how_to_overlap=self.overlap,
            config=self._model.config,
            # skip_pruned_components=self._pruned_components,
        )

        # prune the model (fake, just nullify the mask)
        if "attn_heads" in self.components:
            for layer, heads in components_to_prune.attention_heads_to_prune.items():
                self._fake_pruning_masks[f"attn_heads"][layer, heads] = 0.0
        if "attn_layers" in self.components:
            layers = components_to_prune.attention_layers_to_prune
            self._fake_pruning_masks[f"attn_layers"][layers] = 0.0
        if "ffn_neurons" in self.components:
            for layer, neurons in components_to_prune.ffn_neurons_to_prune.items():
                self._fake_pruning_masks[f"ffn_neurons"][layer, neurons] = 0.0
        if "ffn_layers" in self.components:
            layers = components_to_prune.ffn_layers_to_prune
            self._fake_pruning_masks[f"ffn_layers"][layers] = 0.0
        if "hidden_states" in self.components:
            states = components_to_prune.hidden_states_to_prune
            self._fake_pruning_masks[f"hidden_states"][states] = 0.0

        # update _pruned_components, extend the pruned components
        for layer, heads in components_to_prune.attention_heads_to_prune.items():
            self._pruned_components.attention_heads_to_prune[layer].extend(heads)
        self._pruned_components.attention_layers_to_prune.extend(components_to_prune.attention_layers_to_prune)
        for layer, neurons in components_to_prune.ffn_neurons_to_prune.items():
            self._pruned_components.ffn_neurons_to_prune[layer].extend(neurons)
        self._pruned_components.ffn_layers_to_prune.extend(components_to_prune.ffn_layers_to_prune)
        self._pruned_components.hidden_states_to_prune.extend(components_to_prune.hidden_states_to_prune)

        # log the pruning
        self._current_pruned_ratio += to_prune

        if self.neptune_run:
            for name, value in components_info._asdict().items():
                self.neptune_run[f"pruning/step_{step_id}_{name}"].upload(File.as_pickle(tensor_to_list(value)))
            for name, value in components_importance._asdict().items():
                self.neptune_run[f"pruning/step_{step_id}_{name}"].upload(File.as_pickle(tensor_to_list(value)))

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
        print("ON TRAIN BEGIN")
        assert "model" in kwargs, "The model and tokenizer have to be passed to the callback"
        self._init_pruning_params(args, state, kwargs["model"])

        # prune the model at the beginning (0th step)
        if 0 in self._pruning_step_to_share:
            to_prune = self._pruning_step_to_share[0]
            round_to = self.round_to if 0 == self._max_pruning_step else 1
            # verbose the pruning
            print(
                f"PRUNING: step 0/{state.max_steps} (max {self._max_pruning_step}), "
                f"pruning now {to_prune:.1%} (total {self._current_pruned_ratio + to_prune:.1%}), round to {round_to}"
            )
            # prune the model
            self._fake_prune_model(to_prune, round_to, step_id=0)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        print("ON STEP BEGIN")
        # check if it is time to prune (0th step is pruned in on_train_begin)
        if state.global_step in self._pruning_step_to_share and state.global_step != 0:
            # get the percent to prune this round
            to_prune = self._pruning_step_to_share[state.global_step]
            # if it is the last round, round_to the target ratio
            round_to = self.round_to if state.global_step == self._max_pruning_step else 1

            # verbose the pruning
            print(
                f"PRUNING: step {state.global_step}/{state.max_steps} (max {self._max_pruning_step}), "
                f"pruning now {to_prune:.1%} (total {self._current_pruned_ratio + to_prune:.1%}), round to {round_to}"
            )

            # prune the model
            self._fake_prune_model(to_prune, round_to, step_id=state.global_step)

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
        self._fake_pruning_hooks = []

        # update saved pruned components to match round_to

    def __del__(self) -> None:
        # remove the hooks
        for hook in self._fake_pruning_hooks:
            hook.remove()
        self._fake_pruning_hooks = []
