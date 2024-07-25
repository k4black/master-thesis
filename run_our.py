from __future__ import annotations

import gc
import shutil
from typing import Optional

import neptune
import torch
import typer
from peft import LoraConfig, LoraModel, PeftModel, PeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.integrations import NeptuneCallback

from adaptive_pruning.pruning import (
    prune_attention_heads,
    prune_attention_layers,
    prune_ffn_layers,
    prune_ffn_neurons,
    prune_hidden_states,
)
from adaptive_pruning.trainer_callback import PruningTrainerCallback
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
    pruning_dataset: str = "bookcorpus",
    batch_size: int = 32,
    how_to_collect: str = "grads",  # grads or activations or random
    how_to_average: str = "fisher_info",  # fisher_info, sum, mean, max or entropy
    how_to_overlap: str = "fixed",  # fixed, relative, meta
    pruning_components: str = "attn_heads+ffn_neurons",  # attn_heads, attn_layers, ffn_neurons, ffn_layers, hidden_states, all
    pruning_ratio: float = 0.5,
    num_samples: int = 256,
    num_iterations: int = 1,
    num_train_epochs: float = 0.1,
    prune_before_training: bool = False,
    round_to: int = 1,
    seed: int = 42,
    evaluate_on: Optional[str] = "perplexity+short+bias",
    save_model_as: Optional[str] = None,
    extra_tags: Optional[str] = None,  # split by +
) -> None:
    set_random_seed(seed)

    if pruning_components == "all":
        pruning_components = "attn_heads+attn_layers+ffn_neurons+ffn_layers+hidden_state"
    pruning_components_list: list[str] = pruning_components.split("+")
    assert len(pruning_components), "Need to select at least one pruning method"

    is_uniform = False
    if "attn_heads_uniform" in pruning_components_list:
        pruning_components_list.remove("attn_heads_uniform")
        pruning_components_list.append("attn_heads")
        is_uniform = True
    if "ffn_neurons_uniform" in pruning_components_list:
        pruning_components_list.remove("ffn_neurons_uniform")
        pruning_components_list.append("ffn_neurons")
        is_uniform = True

    # setup logging
    neptune_run = create_neptune_run(
        base_model=base_model,
        lib="our",
        pruning_ratio=pruning_ratio,
        pruning_components=pruning_components_list,
        pruning_round_to=round_to,
        num_iterations=num_iterations,
        calibration_dataset=pruning_dataset,
        calibration_batch_size=batch_size,
        calibration_num_samples=num_samples,
        calibration_how_to_collect=how_to_collect,
        calibration_how_to_average=how_to_average,
        calibration_how_to_overlap=how_to_overlap,
        prune_before_training=prune_before_training,
        attention_type=attention_type,
        save_model_as=save_model_as,
        finetuning=num_train_epochs > 0,
        finetuning_epochs=num_train_epochs,
        finetuning_dataset="alpaca-gpt4",
        extra_tags=["our"] if not extra_tags else ["our"] + extra_tags.split("+"),
    )

    # Load the finetuned model and the corresponding tokenizer
    config, model, tokenizer = load_llama_model(
        base_model, attention_type=attention_type, device="cuda" if IS_CUDA_AVAILABLE else "cpu"
    )

    original_model_stats, original_model_size = measure_model_stats(model, tokenizer, print_results=False)

    # load dataset
    print(f"Loading dataset {pruning_dataset}...")
    calibration_dataset = get_tokenized_dataset(pruning_dataset, "train", tokenizer, num_samples, 128, streaming=True)
    train_dataset = get_tokenized_dataset("alpaca-gpt4", "train", tokenizer, streaming=False, seq_len=512)
    validation_dataset = get_tokenized_dataset(
        "wikitext2", "validation", tokenizer, seq_len=512, streaming=False, n_samples=1000, drop_empty_strings=True
    )
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)

    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"Calibration dataloader: {len(calibration_dataset)} samples")
    print("Dataset loaded")

    # print model with sample input
    print(model)

    print("-" * 80)

    # Collect components info
    # if how_to_collect == "grads":
    #     components_info = collect_mask_gradients(model, calibration_dataloader)
    # elif how_to_collect == "full_grads":
    #     components_info = collect_full_gradients(model, calibration_dataloader)
    # elif how_to_collect == "activations":
    #     components_info = collect_activations(model, calibration_dataloader)
    # elif how_to_collect == "weights":
    #     components_info = collect_weight_magnitudes(model)
    # elif how_to_collect == "random":
    #     components_info = collect_random_numbers(model)
    # else:
    #     assert False, f"Unknown how_to_collect: {how_to_collect}"

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
        output_dir="./results/our",
        report_to="none",
        learning_rate=1e-4,
        weight_decay=0.001,
        warmup_ratio=0.05,
        gradient_accumulation_steps=1,
        # max_grad_norm=0.3,
        num_train_epochs=num_train_epochs,
        optim="adamw_torch",
        auto_find_batch_size=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        group_by_length=True,
        logging_steps=100,
        eval_steps=200 if validation_dataset else None,
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=False,
        fp16=IS_CUDA_AVAILABLE,
        fp16_full_eval=IS_CUDA_AVAILABLE,
        use_mps_device=False,
        eval_strategy="steps",
        save_strategy="steps",
        eval_on_start=True,
        seed=seed,
    )
    pruning_callback = PruningTrainerCallback(
        target_ratio=pruning_ratio,
        components=pruning_components_list,
        strategy=how_to_collect,
        average=how_to_average,
        overlap=how_to_overlap,
        data_loader=calibration_dataloader,
        is_uniform=is_uniform,
        round_to=round_to,
        num_epochs=num_train_epochs,  # TODO: select different epochs for pruning
        num_iterations=num_iterations,
        neptune_run=neptune_run,
        prune_before_training=prune_before_training,
    )
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[pruning_callback, NeptuneCallback(run=neptune_run)],
        data_collator=collate_fn,
    )

    print("\n==================TRAINING + FIND PRUNING==================\n")
    # Train the model
    trainer.train()
    # Re-run neptune run as Trainer stops it
    neptune_run = neptune.init_run(with_id=neptune_run._sys_id)

    print("\n==================SAVE LOAD MODEL==================\n")
    # Save the trained adapters
    model = model.to(device="cpu", dtype=torch.float32)
    del trainer

    # merge the LoRA adapters if any
    if isinstance(model, PeftModel):
        model.merge_and_unload()
        if isinstance(model, PeftModelForCausalLM):
            model = model.base_model
        if isinstance(model, LoraModel):
            model = model.model
        assert not isinstance(model, PeftModel)

    model.save_pretrained(f"results/our-{neptune_run._sys_id}")
    if IS_CUDA_AVAILABLE:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        f"results/our-{neptune_run._sys_id}",
        config=config,
        torch_dtype=torch.float32 if not IS_CUDA_AVAILABLE else torch.float16,
    )
    model = model.to(device="cuda" if IS_CUDA_AVAILABLE else "cpu")
    # delete the old model path
    shutil.rmtree(f"results/our-{neptune_run._sys_id}")

    print("\n==================ACTUAL PRUNING==================\n")

    prune_attention_layers(model, pruning_callback._pruned_components.attention_layers_to_prune)
    prune_attention_heads(model, pruning_callback._pruned_components.attention_heads_to_prune)
    prune_ffn_layers(model, pruning_callback._pruned_components.ffn_layers_to_prune)
    prune_ffn_neurons(model, pruning_callback._pruned_components.ffn_neurons_to_prune)
    prune_hidden_states(model, pruning_callback._pruned_components.hidden_states_to_prune)

    print("\n==================AFTER TRAINING==================\n")
    print("Model after training:")
    print(model)

    print("-" * 80)
    if IS_CUDA_AVAILABLE:
        # TODO: fix, next(model.parameters()).dtype float16, but error as full precision
        model = model.to(dtype=torch.float16)
    else:
        model = model.to(dtype=torch.float32)
    pruned_model_stats, pruned_model_size = measure_model_stats(
        model, tokenizer, original_model_stats, print_results=True
    )
    neptune_record_pruned_model(
        neptune_run, original_model_stats, original_model_size, pruned_model_stats, pruned_model_size
    )

    if save_model_as:
        save_model_tokenizer(model, tokenizer, "results/" + save_model_as, neptune_run=neptune_run)

    # Log pruned model
    if evaluate_on:
        print("\n==================Evaluation after Pruning==================\n")
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
