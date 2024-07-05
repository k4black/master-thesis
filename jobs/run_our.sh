nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


base_model_name='huggyllama/llama-7b'
save_as='our-llama-7b-pruned'

pruning_rations="0.05 0.1 0.2 0.3 0.4 0.5"

for pruning_ratio in $pruning_ratios; do
    echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

    echo "[START] - Start Pruning Model ($pruning_ratio)"
    python run_our.py \
        --base-model=$base_model_name \
        --pruning-components "attn_heads+ffn_neurons" \
        --pruning-ratio $pruning_ratio \
        --batch-size 8 \
        --how-to-collect "grads" \
        --how-to-average "fisher_info" \
        --how-to-overlap "fixed" \
        --evaluate-on="perplexity+full+bias"
    echo "[END] - Finish Pruning Model ($pruning_ratio)"
done
