import os
import copy
from pathlib import Path
import typing

import click
import neptune
import torch
from transformers import LlamaTokenizer
# from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention

if typing.TYPE_CHECKING:
    import external.llm_pruner.LLMPruner.torch_pruning as tp
    from external.llm_pruner.LLMPruner.pruner import hf_llama_pruner as llama_pruner
    from external.llm_pruner.LLMPruner.utils.logger import LoggerWithDepth
    from external.llm_pruner.LLMPruner.datasets.example_samples import get_examples
    from external.llm_pruner.LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
else:
    # add external.llm_pruner to access LLMPruner
    os.sys.path.append((Path(__file__).parent / "external" / "llm_pruner").as_posix())
    import LLMPruner.torch_pruning as tp
    from LLMPruner.pruner import hf_llama_pruner as llama_pruner
    from LLMPruner.utils.logger import LoggerWithDepth
    from LLMPruner.datasets.example_samples import get_examples
    from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP
from utils import set_random_seed, evaluate_model


@click.command()
@click.option(
    "--base_model",
    type=str,
    default="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
    help="base model name",
)
@click.option(
    "--save_ckpt_log_name",
    type=str,
    default="TinyLlama-1.1B-pruned",
    help="the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}",
)
@click.option("--pruning_ratio", type=float, default=0.5, help="pruning ratio")
@click.option("--pruner_type", type=str, default="l2", help="pruner type")
@click.option("--max_seq_len", type=int, default=128, help="max sequence length")
@click.option("--channel_wise", is_flag=True, help="channel wise")
@click.option("--block_wise", is_flag=True, help="block wise")
@click.option("--layer_wise", is_flag=True, help="layer wise")
@click.option("--layer", type=int, default=12, help="remain the previous n layers")
@click.option("--block_attention_layer_start", type=int, help="start layer of block attention layers", default=3)
@click.option("--block_attention_layer_end", type=int, help="end layer of block attention layers", default=31)
@click.option("--block_mlp_layer_start", type=int, help="start layer of block mlp layers", default=3)
@click.option("--block_mlp_layer_end", type=int, help="end layer of block mlp layers", default=31)
@click.option("--iterative_steps", type=int, default=1, help="Iteration step for pruning. Default=1")
@click.option("--grouping_strategy", type=str, default="sum", help="Reduce method for grouping")
@click.option("--global_pruning", is_flag=True, help="whether global pruning")
@click.option(
    "--taylor", type=str, default="param_first", help="choose from [vectorize, param_second, param_first, param_mix]"
)
@click.option("--num_examples", type=int, default=10)
@click.option("--device", type=str, default="cpu", help="device")
@click.option("--eval_device", type=str, default="cpu", help="eval device")
@click.option("--seed", type=int, default=42, help="seed")
@click.option("--save_model", is_flag=True, help="if save model")
@click.option("--evaluate", is_flag=True, help="if evaluate")
def main(
    base_model: str,
    save_ckpt_log_name: str,
    pruning_ratio: float,
    pruner_type: str,
    max_seq_len: int,
    channel_wise: bool,
    block_wise: bool,
    layer_wise: bool,
    layer: int,
    block_attention_layer_start: int,
    block_attention_layer_end: int,
    block_mlp_layer_start: int,
    block_mlp_layer_end: int,
    iterative_steps: int,
    grouping_strategy: str,
    global_pruning: bool,
    taylor: str,
    num_examples: int,
    device: str,
    eval_device: str,
    seed: int,
    save_model: bool,
    evaluate: bool,
):
    set_random_seed(seed)

    # setup logging
    neptune_run = neptune.init_run(tags=['llm-pruner', base_model, 'block_wise' if block_wise else 'channel_wise' if channel_wise else 'layer_wise'])
    method = [
        "attn-heads+ffn-neurons" if block_wise else None,
        "hidden-state" if channel_wise else None,
        "attn-layers+ffn-layers" if layer_wise else None,
        pruner_type,
    ]
    method = "+".join([m for m in method if m ])
    neptune_run["parameters"] = {
        "base_model": base_model,
        "lib": "llm-pruner",
        "method": method,
        "ratio": pruning_ratio,
        "pruning_components": "attn-heads+ffn-neurons" if block_wise else "hidden-state" if channel_wise else "attn-layers+ffn-layers" if layer_wise else "attn-heads+ffn-neurons",
        "how_to_collect": pruner_type,
        "how_to_average": taylor,
        "num_samples": num_examples,
    }

    logger = LoggerWithDepth(
        env_name=f"{save_ckpt_log_name}",
        config={},
        root_dir="results/llm_pruner",
        setup_sublogger=True,
    )

    logger.log(f"Loading model {base_model}...")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    if device != "cpu":
        model.half()
    model.to(device)

    pruner_type = pruner_type.lower()
    assert pruner_type in ["random", "l2", "l1", "taylor"]

    for param in model.parameters():
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Only for building the dependency graph.
    # Any input will be fine since the computation result are not taken into consideration.
    forward_prompts = torch.tensor(
        [
            [1, 306, 4658, 278, 6593, 310, 2834, 338],
            [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
        ]
    ).to(device)
    # print('-'*64)
    # print(model)
    # summary(model, input_data=forward_prompts, depth=4, device=device)
    # for i, layer in enumerate(model.model.layers):
    #     print(f"Layer {i}")
    #     print(f"  Attention: {layer.self_attn.q_proj.weight.shape} (q_proj), {layer.self_attn.k_proj.weight.shape} (k_proj), {layer.self_attn.v_proj.weight.shape} (v_proj), {layer.self_attn.o_proj.weight.shape} (o_proj)")
    #     print(f"  MLP: {layer.mlp.gate_proj.weight.shape} (gate_proj), {layer.mlp.up_proj.weight.shape} (up_proj), {layer.mlp.down_proj.weight.shape} (down_proj)")

    if pruner_type == "random":
        imp = tp.importance.RandomImportance()
    elif pruner_type == "l1":
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == "l2":
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == "taylor":
        imp = llama_pruner.TaylorImportance(group_reduction=grouping_strategy, taylor=taylor)
    else:
        raise NotImplementedError

    logger.log(f"Use {pruner_type} pruner...")

    if block_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": global_pruning,
            "iterative_steps": iterative_steps,
            "ch_sparsity": pruning_ratio,
            "ignored_layers": [],
            "channel_groups": {},
            "consecutive_groups": {layer.self_attn.q_proj: layer.self_attn.head_dim for layer in model.model.layers},
            "customized_pruners": {LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner},
            "root_module_types": None,
            "root_instances": [
                model.model.layers[i].self_attn.q_proj
                for i in range(block_attention_layer_start, block_attention_layer_end)
            ]
            + [model.model.layers[i].mlp.gate_proj for i in range(block_mlp_layer_start, block_mlp_layer_end)],
        }
        logger.log(f"Pruning Attention Layer = {list(range(block_attention_layer_start, block_attention_layer_end))}")
        logger.log(f"Pruning MLP Layer = {list(range(block_mlp_layer_start, block_mlp_layer_end))}")

        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()

        logger.log("Start Pruning")
        for i in range(iterative_steps):
            logger.log(f"Iter {i + 1}/{iterative_steps}")

            if pruner_type in ["taylor"]:
                example_prompts = get_examples("bookcorpus", tokenizer, num_examples, seq_len=64).to(device)
                logger.log(f"Start Backwarding in iterative steps = {i}...")
                if taylor in ["param_mix", "param_second"]:
                    for j in range(num_examples):
                        logger.log(f"Example {j + 1}/{num_examples}")
                        batch_input = example_prompts[j].unsqueeze(0)
                        loss = model(batch_input, labels=batch_input).loss
                        logger.log(f"Loss = {loss}")
                        loss.backward()

                        for module_param in model.parameters():
                            module_param.grad = module_param.grad * module_param.grad / num_examples
                            if hasattr(module_param, "acc_grad"):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                        model.zero_grad()
                        del loss.grad

                loss = model(example_prompts, labels=example_prompts).loss
                logger.log(f"Loss = {loss}")
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log(f"After Iter {i + 1}/{iterative_steps}, #parameters: {after_pruning_parameters}")

            # modify inferece-related attributes
            for layer in model.model.layers:
                layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if "weight" in name:
                module.grad = None

        del pruner

    elif channel_wise:
        kwargs = {
            "importance": imp,
            "global_pruning": global_pruning,
            "iterative_steps": iterative_steps,
            "ch_sparsity": pruning_ratio,  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
            "ignored_layers": [],
            # "round_to": model.config.num_attention_heads * 2,
            "channel_groups": {
                # layer.self_attn: layer.self_attn.num_heads for layer in model.model.layers
            },
            "customized_pruners": {
                LlamaRMSNorm: llama_pruner.hf_rmsnorm_pruner,
                # LlamaAttention: llama_pruner.hf_attention_pruner,
            },
            "root_module_types": [LlamaRMSNorm, LlamaAttention],
        }

        pruner = tp.pruner.MetaPruner(model, forward_prompts, **kwargs)
        model.zero_grad()

        logger.log("Start Pruning")
        assert iterative_steps >= 1
        for i in range(iterative_steps):
            logger.log(f"Iter {i + 1}/{iterative_steps}")

            if pruner_type in ["taylor"]:
                example_prompts = get_examples("bookcorpus", tokenizer, n_samples=num_examples, seq_len=max_seq_len).to(device)
                logger.log(f"Start Backwarding in iterative steps = {i}...")
                loss = model(example_prompts, labels=example_prompts).loss
                logger.log(f"Loss = {loss}")
                loss.backward()

            pruner.step()

            after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.log(
                f"After Iter {i + 1}/{iterative_steps}, #parameters: {after_pruning_parameters}, Ratio = {100.0 * after_pruning_parameters / before_pruning_parameters:.4f}%"
            )

        # Clean the gradient in the model
        model.zero_grad()
        for name, module in model.named_parameters():
            if "weight" in name:
                module.grad = None

        # modify inferece-related attributes
        model.config.hidden_size = model.model.embed_tokens.weight.shape[1]
        model.zero_grad()

        del pruner

    elif layer_wise:
        model.model.layers = model.model.layers[:layer]
        after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    else:
        raise NotImplementedError

    logger.log(
        f"#Param before: {before_pruning_parameters}, #Param after: {after_pruning_parameters}, Ratio = {100.0 * after_pruning_parameters / before_pruning_parameters:.4f}%"
    )

    if save_model:
        logger.log(f"Save the pruned model as {logger.best_checkpoint_path}")
        model.half()
        torch.save(
            {
                "model": model,
                "tokenizer": tokenizer,
            },
            logger.best_checkpoint_path,
        )

    print('-'*64)
    print(model)
    # summary(model, input_data=forward_prompts, depth=4, device=eval_device)
    # go layer by layer and print actual size of model.model.layers[i].self_attn params and model.model.layers[i].mlp params
    # for i, layer in enumerate(model.model.layers):
    #     print(f"Layer {i}")
    #     print(f"  Attention: {layer.self_attn.q_proj.weight.shape} (q_proj), {layer.self_attn.k_proj.weight.shape} (k_proj), {layer.self_attn.v_proj.weight.shape} (v_proj), {layer.self_attn.o_proj.weight.shape} (o_proj)")
    #     print(f"  MLP: {layer.mlp.gate_proj.weight.shape} (gate_proj), {layer.mlp.up_proj.weight.shape} (up_proj), {layer.mlp.down_proj.weight.shape} (down_proj)")

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if evaluate:
        logger.log("\n==================Evaluation after Pruning==================\n")
        eval_results = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            inference_dataset='bookcorpus',
            name=save_ckpt_log_name,
            device=eval_device,
        )
        neptune_run["evaluation"] = eval_results


if __name__ == "__main__":
    main()
