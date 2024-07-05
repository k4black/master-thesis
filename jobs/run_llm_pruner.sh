nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


base_model_name='huggyllama/llama-7b'
save_as='llm-pruner-llama-7b-pruned'

pruning_rations="0.05 0.1 0.2 0.3 0.4 0.5"


for pruning_ratio in $pruning_rations; do
    echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

    echo "[START] - Start Pruning Model ($pruning_ratio)"
    python run_llm_pruner.py \
        --base-model=$base_model_name \
        --pruning-ratio $pruning_ratio \
        --block-wise \
        --pruner-type "taylor" \
        --evaluate-on="perplexity+full+bias"
#        --block-wise \
#        --channel-wise \
    #    --save-model-as TBA
    echo "[END] - Finish Pruning Model ($pruning_ratio)"
done
