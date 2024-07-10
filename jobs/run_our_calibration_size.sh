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

#num_calibration_samples_list="8 16 32 64 128 256 512"
num_calibration_samples_list="512"

for num_calibration_samples in $num_calibration_samples_list; do
    echo "-->>> >>> Num samples: $num_calibration_samples <<< <<<--"

    echo "[START] - Start Pruning Model ($num_calibration_samples)"
    python run_our.py \
        --base-model=$base_model_name \
        --pruning-components "attn_heads+ffn_neurons" \
        --pruning-ratio 0.3 \
        --batch-size 1 \
        --num-samples $num_calibration_samples \
        --how-to-collect "grads" \
        --how-to-average "fisher_info" \
        --how-to-overlap "fixed" \
        --evaluate-on="perplexity+short+bias" \
        --extra-tags "calibration_samples"
    echo "[END] - Finish Pruning Model ($num_calibration_samples)"
done


batch_size_list="4 8 16"
num_calibration_samples=256

for batch_size in $batch_size_list; do
    echo "-->>> >>> Batch size: $batch_size <<< <<<--"

    echo "[START] - Start Pruning Model ($batch_size)"
    python run_our.py \
        --base-model=$base_model_name \
        --pruning-components "attn_heads+ffn_neurons" \
        --pruning-ratio 0.3 \
        --batch-size $batch_size \
        --num-samples $num_calibration_samples \
        --how-to-collect "grads" \
        --how-to-average "fisher_info" \
        --how-to-overlap "fixed" \
        --evaluate-on="perplexity+short+bias" \
        --extra-tags "calibration_samples"
    echo "[END] - Finish Pruning Model ($batch_size)"
done