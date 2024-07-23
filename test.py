import torch
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from adaptive_pruning.trainer_callback import PruningTrainerCallback


class DummyCallback(TrainerCallback):
    """Just print each step of callback"""

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"on_epoch_begin: EPOCH {state.epoch}, STEP {state.global_step}, MAX_STEPS {state.max_steps}")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"on_epoch_end: {state.epoch}")

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("INIT END!")
        print(f"on_init_end: {state.epoch}, {state.global_step}, {state.max_steps}")
        print(state)

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"on_optimizer_step: {state.epoch}")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("STEP BEGIN!")
        print(f"on_step_begin: epoch {state.epoch}, step {state.global_step}, max_steps {state.max_steps}")

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("STEP END!")
        print(f"on_step_end: epoch {state.epoch}, step {state.global_step}, max_steps {state.max_steps}")

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("SUBSTEP END!")
        print(f"on_substep_end: {state.epoch}")

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"on_train_begin: {state.epoch}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print(f"on_train_end: {state.epoch}")


class DummyTinyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(10, 10).to("mps")

    def forward(self, input_ids, labels=None):
        input_ids = input_ids.to("mps")
        labels = labels.to("mps")
        out = self.linear(input_ids).to("mps")
        return {
            "loss": torch.nn.functional.mse_loss(out, labels).to("cpu"),
            "logits": out.to("mps"),
        }


if __name__ == "__main__":
    model = DummyTinyModel()
    model = model.to("mps")
    training_args = TrainingArguments(
        report_to="none",
        output_dir="./results/test",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        dataloader_drop_last=False,
    )
    print("CREATE")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=[{"input_ids": torch.rand(10).to("mps"), "labels": torch.rand(10).to("mps")} for _ in range(112)],
        callbacks=[
            PruningTrainerCallback(
                dataset=[{"input_ids": torch.rand(10).to("mps")} for _ in range(32)],
                target_ratio=0.3,
                components=["ffn-neurons"],
                strategy="mask-grads",
                average="fisher-info",
                overlap="fixed",
                num_epochs=2,
                num_iterations=5,
                round_to=32,
            ),
        ],
    )
    print("TRAIN")
    trainer.train()
