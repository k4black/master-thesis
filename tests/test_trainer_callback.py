import copy

import pytest
from datasets import Dataset
from transformers import PretrainedConfig, PreTrainedModel, TrainerControl, TrainerState, TrainingArguments, \
    DataCollatorWithPadding

from adaptive_pruning.trainer_callback import PruningTrainerCallback


# fixtures: test_config_llama, test_lm_model_llama, random_lm_dataloader


class TestPruningTrainerCallback:

    @pytest.fixture()
    def args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir="./tmp/test_output",
            per_device_train_batch_size=2,
            num_train_epochs=2,
        )

    @pytest.fixture()
    def state(self) -> TrainerState:
        return TrainerState(
            max_steps=200,
            num_train_epochs=2,
            train_batch_size=2,
        )

    @pytest.fixture()
    def control(self) -> TrainerControl:
        return TrainerControl()

    @pytest.mark.parametrize("num_iterations", [1, 2, 4, 16, 32])
    def test_simple_prune_before_training(
        self,
        test_config_llama: PretrainedConfig,
        test_lm_model_llama: PreTrainedModel,
        random_lm_dataset: Dataset,
        lm_data_collator: DataCollatorWithPadding,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        num_iterations: int,
    ) -> None:
        total_heads = test_config_llama.num_hidden_layers * test_config_llama.num_attention_heads
        total_neurons = test_config_llama.num_hidden_layers * test_config_llama.intermediate_size
        total_states = test_config_llama.hidden_size

        callback = PruningTrainerCallback(
            target_ratio=0.5,
            components=["attn_heads", "ffn_neurons", "hidden_states"],
            strategy="grads",
            average="fisher_info",
            overlap="fixed",
            dataset=random_lm_dataset,
            data_collator=lm_data_collator,
            num_samples=8,
            batch_size=4,
            num_iterations=num_iterations,
            prune_before_training=True,
            round_to=1,
        )

        # init
        callback.on_init_end(args=args, state=state, control=control, model=test_lm_model_llama)
        assert not callback._pruning_step_to_pruning_ratio

        # training
        callback.on_train_begin(args=args, state=state, control=control, model=test_lm_model_llama)
        assert len(callback._pruning_step_to_pruning_ratio) == num_iterations
        assert sum(len(heads) for heads in callback.pruning_components.attention_heads_to_prune.values()) == int(
            total_heads * 0.5
        )
        assert len(callback.pruning_components.attention_layers_to_prune) == 0
        assert sum(len(layers) for layers in callback.pruning_components.ffn_neurons_to_prune.values()) == int(
            total_neurons * 0.5
        )
        assert len(callback.pruning_components.ffn_layers_to_prune) == 0
        assert len(callback.pruning_components.hidden_states_to_prune) == int(total_states * 0.5)
        copy_pruning_components = copy.deepcopy(callback.pruning_components)

        # training steps
        for i in range(201):
            state.global_step = i
            callback.on_step_begin(args=args, state=state, control=control, model=test_lm_model_llama)

        # end of training
        callback.on_train_end(args=args, state=state, control=control, model=test_lm_model_llama)

        assert callback.pruning_components == copy_pruning_components

    @pytest.mark.parametrize("num_iterations", [1, 2, 4, 16, 32])
    def test_simple_prune_during_training(
        self,
        test_config_llama: PretrainedConfig,
        test_lm_model_llama: PreTrainedModel,
        random_lm_dataset: Dataset,
        lm_data_collator: DataCollatorWithPadding,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        num_iterations: int,
    ) -> None:
        total_heads = test_config_llama.num_hidden_layers * test_config_llama.num_attention_heads
        total_neurons = test_config_llama.num_hidden_layers * test_config_llama.intermediate_size
        total_states = test_config_llama.hidden_size

        callback = PruningTrainerCallback(
            target_ratio=0.5,
            components=["attn_heads", "ffn_neurons", "hidden_states"],
            strategy="grads",
            average="fisher_info",
            overlap="fixed",
            dataset=random_lm_dataset,
            data_collator=lm_data_collator,
            num_samples=8,
            batch_size=4,
            num_iterations=num_iterations,
            prune_before_training=False,
            round_to=1,
        )

        # init
        callback.on_init_end(args=args, state=state, control=control, model=test_lm_model_llama)
        assert not callback._pruning_step_to_pruning_ratio

        # training
        callback.on_train_begin(args=args, state=state, control=control, model=test_lm_model_llama)
        assert callback._pruning_step_to_pruning_ratio
        assert len(callback._pruning_step_to_pruning_ratio) == num_iterations

        # training steps
        for i in range(201):  # TODO: check Trainer actually reach the last max_steps
            state.global_step = i
            callback.on_step_begin(args=args, state=state, control=control, model=test_lm_model_llama)

            # check during training
            if i < callback._max_pruning_step:
                assert sum(len(heads) for heads in callback.pruning_components.attention_heads_to_prune.values()) < int(
                    total_heads * 0.5
                )
                assert len(callback.pruning_components.attention_layers_to_prune) == 0
                assert sum(len(layers) for layers in callback.pruning_components.ffn_neurons_to_prune.values()) < int(
                    total_neurons * 0.5
                )
                assert len(callback.pruning_components.ffn_layers_to_prune) == 0
                assert len(callback.pruning_components.hidden_states_to_prune) < int(total_states * 0.5)

        # check after training
        assert sum(len(heads) for heads in callback.pruning_components.attention_heads_to_prune.values()) == int(
            total_heads * 0.5
        )
        assert len(callback.pruning_components.attention_layers_to_prune) == 0
        assert sum(len(layers) for layers in callback.pruning_components.ffn_neurons_to_prune.values()) == int(
            total_neurons * 0.5
        )
        assert len(callback.pruning_components.ffn_layers_to_prune) == 0
        assert len(callback.pruning_components.hidden_states_to_prune) == int(total_states * 0.5)

        # end of training
        callback.on_train_end(args=args, state=state, control=control, model=test_lm_model_llama)

        assert False

    @pytest.mark.parametrize("num_iterations", [1, 2, 4, 16, 32])
    @pytest.mark.parametrize("round_to", [1, 2, 4, 8, 16])
    @pytest.mark.parametrize("is_uniform", [False, True])
    def test_round_to_pruning(
        self,
        test_config_llama: PretrainedConfig,
        test_lm_model_llama: PreTrainedModel,
        random_lm_dataset: Dataset,
        lm_data_collator: DataCollatorWithPadding,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        num_iterations: int,
        round_to: int,
        is_uniform: bool,
    ) -> None:
        pruning_ratio = 0.768
        kv_group_size = test_config_llama.num_attention_heads // test_config_llama.num_key_value_heads
        total_heads = test_config_llama.num_hidden_layers * test_config_llama.num_attention_heads
        total_neurons = test_config_llama.num_hidden_layers * test_config_llama.intermediate_size
        total_states = test_config_llama.hidden_size

        callback = PruningTrainerCallback(
            target_ratio=pruning_ratio,
            components=["attn_heads", "ffn_neurons", "hidden_states"],
            strategy="grads",
            average="fisher_info",
            overlap="fixed",
            dataset=random_lm_dataset,
            data_collator=lm_data_collator,
            num_samples=8,
            batch_size=4,
            num_iterations=num_iterations,
            prune_before_training=False,
            round_to=round_to,
            is_uniform=is_uniform,
        )

        # init
        callback.on_init_end(args=args, state=state, control=control, model=test_lm_model_llama)
        assert not callback._pruning_step_to_pruning_ratio

        # training
        callback.on_train_begin(args=args, state=state, control=control, model=test_lm_model_llama)
        assert len(callback._pruning_step_to_pruning_ratio) == num_iterations

        # training steps
        for i in range(201):
            state.global_step = i
            callback.on_step_begin(args=args, state=state, control=control, model=test_lm_model_llama)

        # end of training
        callback.on_train_end(args=args, state=state, control=control, model=test_lm_model_llama)

        assert sum(len(heads) for heads in callback.pruning_components.attention_heads_to_prune.values()) <= total_heads
        assert len(callback.pruning_components.attention_layers_to_prune) == 0
        assert sum(len(layers) for layers in callback.pruning_components.ffn_neurons_to_prune.values()) < total_neurons
        assert sum(len(layers) for layers in callback.pruning_components.ffn_neurons_to_prune.values()) % round_to == 0
        assert len(callback.pruning_components.ffn_layers_to_prune) == 0
        assert len(callback.pruning_components.hidden_states_to_prune) < total_states
        assert len(callback.pruning_components.hidden_states_to_prune) % round_to == 0
