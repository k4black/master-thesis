import os
import gc
import copy
from pathlib import Path
import typing

import click
import torch
from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention

from utils import lm_eval_hf_model


@click.command()
@click.option(
    "--base_model",
    type=str,
    default="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
    help="base model name",
)
@click.option(
    "--checkpoint",
    type=str,
    default="TinyLlama-1.1B-pruned",
    help="the path for save the checkpoint, bin with {model, tokenizer} dict",
)
@click.option("--device", type=str, default="cpu", help="device")
@click.option("--tasks", type=str, default="wikitext,piqa,boolq", help="tasks for evaluation")
def main(
    base_model: str,
    checkpoint: str,
    device: str,
    tasks: str,
):
    pruned_dict = torch.load(checkpoint, map_location="cpu")
    tokenizer, model = pruned_dict["tokenizer"], pruned_dict["model"]

    print("\n==================Evaluation==================\n")
    lm_eval_hf_model(
        model=model,
        tokenizer=tokenizer,
        name=checkpoint,
        tasks=tasks.split(","),
        device=device,
        logging_path=Path(checkpoint).with_suffix(".json"),
    )


if __name__ == "__main__":
    main()
