nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


#base_model_name='huggyllama/llama-7b'
base_model_name='meta-llama/Llama-2-7b-hf'
#base_model_name='meta-llama/Meta-Llama-3-8B'
#save_as='llm-pruner-llama-7b-pruned'

pruning_ratios="0.05 0.1 0.2 0.3 0.4 0.5 0.6"
what_to_prune_list="attn_heads+ffn_neurons attn_heads_uniform+ffn_neurons_uniform"


for what_to_prune in $what_to_prune_list; do
    echo "-->>> >>> Pruning Components: $what_to_prune <<< <<<--"

    for pruning_ratio in $pruning_ratios; do
        echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

        echo "[START] - Start Pruning Model ($pruning_ratio)"
        python run_our.py \
            --base-model=$base_model_name \
            --pruning-components $what_to_prune \
            --pruning-ratio $pruning_ratio \
            --batch-size 8 \
            --how-to-collect "grads" \
            --how-to-average "fisher_info" \
            --how-to-overlap "fixed" \
            --evaluate-on="perplexity+short+bias"
        echo "[END] - Finish Pruning Model ($pruning_ratio)"
    done

done
