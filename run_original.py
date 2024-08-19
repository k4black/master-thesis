from __future__ import annotations

import gc
import shutil
from typing import Optional

import neptune
import torch
import typer
from peft import LoraConfig, LoraModel, PeftModel, PeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, \
    BitsAndBytesConfig
from transformers.integrations import NeptuneCallback
from trl import SFTTrainer

from adaptive_pruning.utils import measure_model_stats
from utils import (
    create_neptune_run,
    evaluate_model,
    get_tokenized_dataset,
    load_llama_model,
    neptune_record_pruned_model,
    save_model_tokenizer,
    set_random_seed,
)


IS_CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# disable mac mps
torch.backends.mps.is_built = lambda: False
torch.backends.mps.is_available = lambda: False


def main(
    # base_model: str = "huggyllama/llama-7b",
    base_model: str = "TinyLlama/TinyLlama_v1.1",
    attention_type: Optional[str] = "sdpa",
    pytorch_compile: bool = False,
    num_train_epochs: float = 0.1,
    train_batch_size: int = 2,
    learning_rate: float = 1e-4,
    training_dtype: str = "fp16",  # int8, int4
    seed: int = 42,
    evaluate_on: Optional[str] = "perplexity+short+bias",
    save_model_as: Optional[str] = None,
    extra_tags: Optional[str] = None,
) -> None:
    set_random_seed(seed)

    # setup logging
    neptune_run = create_neptune_run(
        base_model=base_model,
        lib="original",
        pruning_ratio=0.0,
        pruning_components=[],
        num_iterations=0,
        calibration_dataset="",
        calibration_batch_size=0,
        calibration_num_samples=0,
        calibration_how_to_collect="",
        calibration_how_to_average="",
        calibration_how_to_overlap="",
        attention_type=attention_type,
        save_model_as=save_model_as,
        pytorch_compile=pytorch_compile,
        finetuning=num_train_epochs > 0,
        finetuning_epochs=num_train_epochs,
        finetuning_learning_rate=learning_rate,
        finetuning_batch_size=train_batch_size,
        finetuning_dataset="alpaca-gpt4",
        finetuning_dtype=training_dtype if num_train_epochs > 0 else '-',
        extra_tags=["original", "baseline", *extra_tags.split(",")] if extra_tags else ["original", "baseline"],
    )

    # Load the finetuned model and the corresponding tokenizer
    config, model, tokenizer = load_llama_model(
        base_model,
        attention_type=attention_type,
        device="cpu",
    )
    original_model_stats, original_model_size = measure_model_stats(model, tokenizer, print_results=False)
    if num_train_epochs > 0:
        if training_dtype not in ["fp16", "fp32"]:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=training_dtype == "int8",
                load_in_4bit=training_dtype == "int4",
            )
        else:
            quantization_config = None
        config, model, tokenizer = load_llama_model(
            base_model,
            attention_type=attention_type,
            device="cuda" if IS_CUDA_AVAILABLE else "cpu",
            quantization_config=quantization_config,
            train_dtype=training_dtype,
        )

    # load data
    train_dataset = get_tokenized_dataset("alpaca-gpt4", "train", tokenizer, streaming=False, seq_len=256)
    validation_dataset = get_tokenized_dataset(
        "wikitext2", "validation", tokenizer, seq_len=256, streaming=False, n_samples=1000, drop_empty_strings=True
    )
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)

    # print model with sample input
    print(model)

    # Train the model
    print("\n==================SETUP TRAINER==================\n")
    if num_train_epochs > 0:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights="olora",
        )
        # from peft import get_peft_model
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config, mixed=False)
        model.print_trainable_parameters()

    trainer_args = TrainingArguments(
        output_dir="./results/original",
        report_to="none",
        learning_rate=learning_rate,
        weight_decay=0.01,
        # warmup_steps=100,
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        gradient_accumulation_steps=64 // train_batch_size or 1,
        # max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        optim="adamw_torch",
        auto_find_batch_size=True,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=2*train_batch_size,
        # group_by_length=True,
        logging_steps=100,
        eval_steps=200 if validation_dataset else None,
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=False,
        bf16=IS_CUDA_AVAILABLE and "bf16" in training_dtype,
        fp16=IS_CUDA_AVAILABLE and "fp16" in training_dtype,
        # bf16_full_eval=IS_CUDA_AVAILABLE and "bf16" in training_dtype,
        # fp16_full_eval=IS_CUDA_AVAILABLE and "fp16" in training_dtype,
        use_mps_device=False,
        eval_strategy="steps",
        save_strategy="steps",
        eval_on_start=True,
        seed=seed,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[NeptuneCallback(run=neptune_run)],
        data_collator=collate_fn,
    )

    print("\n==================TRAINING==================\n")
    # Train the model
    trainer.train()
    # Re-run neptune run as Trainer stops it
    neptune_run = neptune.init_run(with_id=neptune_run._sys_id)

    print("\n==================MERGE MODEL==================\n")
    # Save the trained adapters
    # model = model.to(device="cpu", dtype=torch.float32)
    # model = model.to(device="cpu")
    model.zero_grad(set_to_none=True)

    # save model or adapters
    model.save_pretrained(f"results/original-{neptune_run._sys_id}")

    del model
    del trainer

    if IS_CUDA_AVAILABLE:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    model = load_llama_model(
        base_model,
        attention_type=attention_type,
        device="cpu",
    )[1]
    if num_train_epochs > 0:
        model = PeftModel.from_pretrained(model=model, model_id=f"results/original-{neptune_run._sys_id}", is_trainable=False)
        # shutil.rmtree(f"results/original-{neptune_run._sys_id}")

        # merge the LoRA adapters
        if isinstance(model, PeftModel):
            if num_train_epochs > 0:
                model.merge_and_unload()
            else:
                model.unload()
            if isinstance(model, PeftModelForCausalLM):
                model = model.base_model
            if isinstance(model, LoraModel):
                model = model.model
            assert not isinstance(model, PeftModel)
    model = model.to(device="cuda" if IS_CUDA_AVAILABLE else "cpu", dtype=torch.float16)

    print("\n==================STATS==================\n")
    pruned_model_stats, pruned_model_size = measure_model_stats(
        model, tokenizer, original_model_stats, print_results=True
    )
    neptune_record_pruned_model(neptune_run, original_model_stats, original_model_size, None, None)

    if save_model_as:
        save_model_tokenizer(model, tokenizer, "results/" + save_model_as, neptune_run=neptune_run)

    if pytorch_compile:
        print("Compiling the model...")
        model = torch.compile(model)

    # Log pruned model
    if evaluate_on:
        print("\n==================EVALUATION==================\n")
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            task_groups=evaluate_on,
            device="cuda" if IS_CUDA_AVAILABLE else "cpu",
        )
        neptune_run["evaluation"] = eval_results

    neptune_run.stop()


if __name__ == "__main__":
    typer.run(main)
