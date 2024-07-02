from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback, TrainerCallback,
)
from datasets import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from fvcore.nn import FlopCountAnalysis
from peft import get_peft_model, LoraConfig

from adaptive_pruning.dataset import load_glue_dataset, measure_glue_metric
from adaptive_pruning.pruning import (
    prune_attention_heads,
    prune_attention_layers,
    prune_ffn_neurons,
    prune_ffn_layers,
    do_prune_hidden_state,
    select_to_prune_attention_heads,
    select_to_prune_attention_layers,
    select_to_prune_ffn_neurons,
    select_to_prune_ffn_layers,
)
from adaptive_pruning.importance import (
    ComponentsImportance,
    ComponentsInfo,
    collect_random_numbers,
    collect_activations,
    collect_weight_magnitudes,
    collect_mask_gradients,
    info_to_mean,
    info_to_max,
    info_to_fisher,
    info_to_entropy, info_to_minus_entropy,
)
from adaptive_pruning.utils import count_total_parameters, format_number

CUDA_DEVICES = (
    [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
)
IS_TPU_AVAILABLE = torch.cuda.is_available() and "TPU" in torch.cuda.get_device_name(0)
IS_CUDA_AVAILABLE = torch.cuda.is_available() and not IS_TPU_AVAILABLE
IS_FP16_AVAILABLE = IS_CUDA_AVAILABLE and not torch.backends.cuda.matmul.allow_tf32
IS_BF16_AVAILABLE = IS_CUDA_AVAILABLE and torch.backends.cuda.matmul.allow_tf32
print(f"CUDA_DEVICES: {CUDA_DEVICES}")
print(f"CUDA_AVAILABLE: {IS_CUDA_AVAILABLE}")
print(f"FP16_AVAILABLE: {IS_FP16_AVAILABLE}")
print(f"BF16_AVAILABLE: {IS_BF16_AVAILABLE}")

# fix backend for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def finetune_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    data_collator: DataCollatorWithPadding,
    compute_metrics: callable | None,
    sample_dataset: Dataset,
    validate_dataset: Dataset,
    batch_size: int = 32,
    max_epochs: int = 4,
    learning_rate: float = 1e-5,
    seed: int = 42,
    lora: bool = True,
) -> None:
    for param in model.parameters():
        param.requires_grad_(True)
    model.train()

    if lora:
        peft_config = LoraConfig(
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="results",
        report_to=[],
        do_eval=True,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        auto_find_batch_size=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=max_epochs,
        warmup_steps=512,
        use_cpu=not IS_CUDA_AVAILABLE,
        fp16=IS_FP16_AVAILABLE,
        fp16_full_eval=IS_FP16_AVAILABLE,
        logging_strategy="epoch",
        evaluation_strategy='epoch',
        save_strategy='epoch',
        # logging_steps=logging_steps,
        # eval_steps=logging_steps,
        # save_steps=logging_steps,
        metric_for_best_model=f"eval_loss",
        # greater_is_better=False,
        load_best_model_at_end=True,
        # logging_first_step=True,
        save_total_limit=1,
        push_to_hub=False,
        seed=seed,
        save_only_model=True,
    )

    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                control.should_evaluate = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sample_dataset,
        eval_dataset=validate_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), EvaluateFirstStepCallback()],
    )

    # trainer.evaluate()
    trainer.train()

    # get best model
    model = trainer.state.best_model

    for param in model.parameters():
        param.requires_grad_(False)

    return model


@click.command()
@click.option("--model_name", type=str, default="bert-base-uncased")
@click.option("--task_name", type=str, default="mnli")
@click.option(
    "--pretrained_model_name_or_path",
    type=str,
    default="JeremiahZ/bert-base-uncased-mnli",
)
@click.option("--batch_size", type=int, default=32)
@click.option("--how_to_collect", type=str, default="grads")  # grads or activations or random
@click.option("--how_to_average", type=str, default="fisher_info")  # fisher_info or mean or entropy
@click.option("--num_samples", type=int, default=256)
@click.option("--num_valid_samples", type=int, default=None)
@click.option("--do_prune_attention_heads", is_flag=True, default=False)
@click.option("--do_prune_attention_heads_uniform", is_flag=True, default=False)
@click.option("--do_prune_attention_layers", is_flag=True, default=False)
@click.option("--do_prune_ffn_neurons", is_flag=True, default=False)
@click.option("--do_prune_ffn_neurons_uniform", is_flag=True, default=False)
@click.option("--do_prune_ffn_layers", is_flag=True, default=False)
@click.option("--do_prune_hidden_state", is_flag=True, default=False)
@click.option("--do_full_finetuning", is_flag=True, default=False)
@click.option("--do_lora_finetuning", is_flag=True, default=False)
@click.option("--use_cache", is_flag=True, default=True)
@click.option("--seed", type=int, default=0)
def main(
    model_name: str,
    task_name: str,
    pretrained_model_name_or_path: str,
    batch_size: int,
    how_to_collect: str,
    how_to_average: str,
    num_samples: int,
    num_valid_samples: int | None,
    do_prune_attention_heads: bool,
    do_prune_attention_heads_uniform: bool,
    do_prune_attention_layers: bool,
    do_prune_ffn_neurons: bool,
    do_prune_ffn_neurons_uniform: bool,
    do_prune_ffn_layers: bool,
    do_prune_hidden_state: bool,
    do_full_finetuning: bool,
    do_lora_finetuning: bool,
    use_cache: bool,
    seed: int,
) -> None:
    assert not (do_full_finetuning and do_lora_finetuning), "Can't do both full finetuning and LoRA finetuning"
    assert (
        do_prune_attention_heads
        or do_prune_attention_heads_uniform
        or do_prune_attention_layers
        or do_prune_ffn_neurons
        or do_prune_ffn_neurons_uniform
        or do_prune_ffn_layers
        or do_prune_hidden_state
    ), "Need to select at least one pruning method"
    assert not (do_prune_attention_heads and do_prune_attention_heads_uniform), "Can't do both head and uniform head"
    assert not (do_prune_ffn_neurons and do_prune_ffn_neurons_uniform), "Can't do both neuron and uniform neuron"

    # Load the finetuned model and the corresponding tokenizer
    print(f"Loading model {pretrained_model_name_or_path}...")
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, config=config)
    if IS_CUDA_AVAILABLE:
        model = model.to("cuda", non_blocking=True)
    for param in model.parameters():
        param.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    print(f"Number of parameters: {format_number(count_total_parameters(model))}")
    print("Model loaded")

    # load dataset
    print(f"Loading dataset {task_name}...")
    dataset = load_glue_dataset(task_name, tokenizer)
    collate_fn = DataCollatorWithPadding(tokenizer)
    sample_dataset = (
        dataset["train"].shuffle(seed=seed).select(range(num_samples))
        if num_samples and len(dataset["train"]) > num_samples
        else dataset["train"]
    )
    validate_dataset = (
        dataset["validation"].shuffle(seed=seed).select(range(num_valid_samples))
        if num_valid_samples and len(dataset["validation"]) > num_valid_samples
        else dataset["validation"]
    )
    print(dataset)

    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"Sample dataloader: {len(sample_dataset)} samples")

    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"Validation dataloader: {len(validate_dataset)} samples")
    print("Dataset loaded")

    # print model with sample input
    # summary(model, input_size=(batch_size, 512), dtypes=['torch.IntTensor'], depth=7, device=model.device)
    print(model)

    print("-" * 80)

    components_info = None
    dataset_model_collect_hash = "info_" + "dataset" + str(sample_dataset._fingerprint) + "_" + pretrained_model_name_or_path.replace("/", "__") + "_" + how_to_collect
    if use_cache and Path(f"results/{dataset_model_collect_hash}.pickle").exists():
        print(f"Loading cached {how_to_collect} from {dataset_model_collect_hash}.pickle...")
        components_info = pickle.load(open(f"results/{dataset_model_collect_hash}.pickle", "rb"))

    if components_info is None:
        if how_to_collect == "grads":
            components_info = collect_mask_gradients(model, sample_dataloader)
        elif how_to_collect == "activations":
            components_info = collect_activations(model, sample_dataloader)
        elif how_to_collect == "weights":
            components_info = collect_weight_magnitudes(model)
        elif how_to_collect == "random":
            components_info = collect_random_numbers(model)
        else:
            assert False, f"Unknown how_to_collect: {how_to_collect}"

        print(f"Saving {how_to_collect} to {dataset_model_collect_hash}.pickle...")
        pickle.dump(components_info, open(f"results/{dataset_model_collect_hash}.pickle", "wb"))

    if how_to_average == "fisher_info":
        components_importance = info_to_fisher(components_info)
    elif how_to_average == "mean":
        components_importance = info_to_mean(components_info)
    elif how_to_average == "max":
        components_importance = info_to_max(components_info)
    elif how_to_average == "entropy":
        components_importance = info_to_entropy(components_info)
    elif how_to_average == "minus_entropy":
        components_importance = info_to_minus_entropy(components_info)
    else:
        assert False, f"Unknown how_to_average: {how_to_average}"
    meta_importance = components_importance.meta_importance

    # print average importance
    print("meta_importance", meta_importance)
    print(f"Average importance for {how_to_collect}/{how_to_average}:")
    for layer in range(config.num_hidden_layers):
        print(f"> Layer {layer}:")
        print(f"  Attention Head: {components_importance.attention_heads_importance[layer].mean().item()}")
        print(f"  Attention Layer sum Heads: {components_importance.attention_heads_importance[layer].sum().item()}")
        print(f"  Attention Layer: {components_importance.attention_heads_importance[layer].mean().item()}")
        print(f"  FF Neuron: {components_importance.ffn_neurons_importance[layer].mean().item()}")
        print(f"  FF Layer sum Neurons: {components_importance.ffn_neurons_importance[layer].sum().item()}")
        print(f"  FF Layer: {components_importance.ffn_layers_importance[layer].mean().item()}")
        # print(f"  Hidden State: {hidden_state_importance.mean().item()}")

    # prune
    stats_list = []
    # for remain_percentage in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    for remain_percentage in [1.0, 0.9, 0.7, 0.5, 0.3, 0.1]:
        print("-" * 80)
        print(
            f"Pruning {(1 - remain_percentage) * 100:.0f}% with {remain_percentage * 100:.0f}% remain "
            f"on {how_to_collect}/{how_to_average}..."
        )

        # load fresh model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path, config=config, ignore_mismatched_sizes=True
        )
        for param in model.parameters():
            param.requires_grad_(False)

        attention_layers_to_prune = []
        if do_prune_attention_layers:
            # convert importance to layers to prune
            attention_layers_to_prune = select_to_prune_attention_layers(
                components_importance.attention_layers_importance, 1 - remain_percentage
            )
            # actually prune layers
            prune_attention_layers(model.bert, attention_layers_to_prune)
            # print layers deleted
            total_percent_layers_deleted = len(attention_layers_to_prune) / model.config.num_hidden_layers
            print(
                f"Total: {total_percent_layers_deleted * 100:.2f}% attention layers deleted, "
                f"{(1 - total_percent_layers_deleted) * 100:.2f}% remain"
            )

        if do_prune_attention_heads or do_prune_attention_heads_uniform:
            # convert importance to heads to prune (skip pruned attention layers)
            attention_heads_to_prune = select_to_prune_attention_heads(
                components_importance.attention_heads_importance, 1 - remain_percentage, uniform_among_layers=do_prune_attention_heads_uniform
            )
            attention_heads_to_prune = {
                layer: heads
                for layer, heads in attention_heads_to_prune.items()
                if layer not in attention_layers_to_prune
            }
            # actually prune heads
            prune_attention_heads(model.bert, attention_heads_to_prune)
            # print layers and number of heads deleted
            total_num_heads_deleted = 0
            for layer in range(config.num_hidden_layers):
                num_heads_deleted = (
                    config.num_attention_heads - model.bert.encoder.layer[layer].attention.self.num_attention_heads
                )
                total_num_heads_deleted += num_heads_deleted
                print(f"> Layer {layer}: {num_heads_deleted} attention heads deleted")
            total_percent_heads_deleted = total_num_heads_deleted / (
                config.num_attention_heads * config.num_hidden_layers
            )
            print(
                f"Total: {total_percent_heads_deleted * 100:.2f}% attention heads deleted, "
                f"{(1 - total_percent_heads_deleted) * 100:.2f}% remain"
            )

        if do_prune_ffn_layers:
            # convert importance to layers to prune
            ffn_layers_to_prune = select_to_prune_ffn_layers(
                components_importance.ffn_layers_importance, 1 - remain_percentage
            )
            # actually prune layers
            prune_ffn_layers(model.bert, ffn_layers_to_prune)
            # print layers deleted
            total_percent_layers_deleted = len(ffn_layers_to_prune) / model.config.num_hidden_layers
            print(
                f"Total: {total_percent_layers_deleted * 100:.2f}% ffn layers deleted, "
                f"{(1 - total_percent_layers_deleted) * 100:.2f}% remain"
            )

        if do_prune_ffn_neurons or do_prune_ffn_neurons_uniform:
            # convert importance to neurons to prune (skip pruned ffn layers)
            neurons_to_prune = select_to_prune_ffn_neurons(
                components_importance.ffn_neurons_importance, 1 - remain_percentage, uniform_among_layers=do_prune_ffn_neurons_uniform
            )
            neurons_to_prune = {
                layer: neurons for layer, neurons in neurons_to_prune.items() if layer not in attention_layers_to_prune
            }
            # actually prune neurons
            prune_ffn_neurons(model.bert, neurons_to_prune)
            # print layers and number of neurons deleted in FF
            total_num_neurons_deleted = 0
            for layer in range(config.num_hidden_layers):
                num_neurons_deleted = (
                    config.intermediate_size - model.bert.encoder.layer[layer].intermediate.dense.out_features
                )
                total_num_neurons_deleted += num_neurons_deleted
                print(f"> Layer {layer}: {num_neurons_deleted} neurons deleted")
            total_percent_neurons_deleted = total_num_neurons_deleted / (
                config.intermediate_size * config.num_hidden_layers
            )
            print(
                f"Total: {total_percent_neurons_deleted * 100:.2f}% neurons deleted, "
                f"{(1 - total_percent_neurons_deleted) * 100:.2f}% remain"
            )

        if do_full_finetuning or do_lora_finetuning:
            finetune_model(
                model,
                tokenizer,
                collate_fn,
                None,
                sample_dataset,
                validate_dataset,
                lora=do_lora_finetuning,
                batch_size=batch_size,
                learning_rate=5e-6 if do_full_finetuning else 5e-5,
                seed=seed,
            )

        if IS_CUDA_AVAILABLE:
            model = model.to("cuda", non_blocking=True)

        # measure after metric
        metric, elapsed_time = measure_glue_metric(
            model,
            validate_dataloader,
            task_name,
        )
        model_params_num = count_total_parameters(model)
        flops = FlopCountAnalysis(model, dict(next(iter(validate_dataloader))))
        stats_list.append(
            {
                "task": task_name,
                "base_model": pretrained_model_name_or_path,
                "how_to_collect": how_to_collect,
                "how_to_average": how_to_average if how_to_collect != "random" else "random",
                "num_sample_samples": len(sample_dataset),
                "num_valid_samples": len(validate_dataset),
                "do_prune_attention_heads": do_prune_attention_heads,
                "do_prune_attention_heads_uniform": do_prune_attention_heads_uniform,
                "do_prune_attention_layers": do_prune_attention_layers,
                "do_prune_ffn_neurons": do_prune_ffn_neurons,
                "do_prune_ffn_neurons_uniform": do_prune_ffn_neurons_uniform,
                "do_prune_ffn_layers": do_prune_ffn_layers,
                "do_prune_hidden_state": do_prune_hidden_state,
                "finetune": (
                    "none"
                    if not (do_full_finetuning or do_lora_finetuning)
                    else "full" if do_full_finetuning else "lora"
                ),
                "timestamp": datetime.now().replace(microsecond=0).isoformat(),
                "remain_percentage": remain_percentage,
                "metric": round(metric, 4),
                "elapsed_time": round(elapsed_time, 2),
                "params_num": model_params_num,
                "params_num_str": format_number(model_params_num),
                # "flops": flops.total(),
            }
        )
        print(
            f"AFTER {remain_percentage * 100}% remain on {how_to_collect}/{how_to_average}: "
            f"{task_name} metric: {metric:.4f} {elapsed_time:.2f}s ({len(validate_dataloader.dataset)} samples)"
        )

    # print stats
    # get 1.0 remain_percentage row
    df_stats = pd.DataFrame.from_records(stats_list)
    max_metric_row = df_stats[df_stats["remain_percentage"] == 1.0].iloc[0]
    df_stats["relative_metric"] = round(df_stats["metric"] / max_metric_row["metric"], 2)
    df_stats["relative_elapsed_time"] = round(df_stats["elapsed_time"] / max_metric_row["elapsed_time"], 2)
    df_stats["relative_params_num"] = round(df_stats["params_num"] / max_metric_row["params_num"], 2)
    print("+" * 80)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.expand_frame_repr",
        False,
        "max_colwidth",
        None,
    ):
        print(df_stats)
    # save stats to existing log csv file
    is_existing_log = Path("results/pruning_stats.csv").exists()
    df_stats.to_csv("results/pruning_stats.csv", mode="a", header=not is_existing_log, index=False)


if __name__ == "__main__":
    if IS_FP16_AVAILABLE:
        with torch.cuda.amp.autocast():
            main()
    else:
        main()
