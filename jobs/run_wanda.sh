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

pruning_ratios="0.1 0.2 0.3 0.4 0.5 0.6"

for pruning_ratio in $pruning_ratios; do
    echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

    echo "[START] - Start Pruning Model ($pruning_ratio)"
    python run_wanda.py \
        --base-model=$base_model_name \
        --pruning-ratio $pruning_ratio \
        --sparsity-type "unstructured" \
        --prune-method "wanda" \
        --evaluate-on="perplexity+full+bias"
    #    --save-model-as TBA
    echo "[END] - Finish Pruning Model ($pruning_ratio)"
done

extra_sparsity_types="2:4 4:8"

for sparsity_type in $extra_sparsity_types; do
    echo "-->>> >>> Sparsity Type: $sparsity_type <<< <<<--"

    echo "[START] - Start Pruning Model ($sparsity_type)"
    python run_wanda.py \
        --base-model=$base_model_name \
        --pruning-ratio 0.5 \
        --sparsity-type $sparsity_type \
        --prune-method "wanda" \
        --evaluate-on="perplexity+full+bias"
    #    --save-model-as TBA
    echo "[END] - Finish Pruning Model ($sparsity_type)"
done
