nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


#base_model_name='huggyllama/llama-7b'
base_model_name='meta-llama/Llama-2-7b-hf'
#base_model_name='meta-llama/Meta-Llama-3-8B'

round_to_list="1 127 32 64 128 256 512 1024"

echo "[START] - Start Rounded Model original (round_to=0)"
python run_compare_pruning_rounding.py \
    --base-model=$base_model_name \
    --pruning-ratio=0.0 \
    --is-uniform \
    --extra-tags "compare_pruning_rounding" \
    --round-to=1
echo "[END] - Finish Rounded Model original (round_to=0)"


for round_to in $round_to_list; do
    echo "-->>> >>> Rounding to: $round_to <<< <<<--"

    echo "[START] - Start Rounded Model uniform (round_to=$round_to)"
    python run_our_pruning_rounding.py \
        --base-model=$base_model_name \
        --pruning-ratio=0.3 \
        --is-uniform \
        --extra-tags "compare_pruning_rounding" \
        --round-to=$round_to
    echo "[END] - Finish Rounded Model uniform (round_to=$round_to)"

    echo "[START] - Start Rounded Model no-uniform (round_to=$round_to)"
    python run_our_pruning_rounding.py \
        --base-model=$base_model_name \
        --pruning-ratio=0.3 \
        --no-is-uniform \
        --extra-tags "compare_pruning_rounding" \
        --round-to=$round_to
    echo "[END] - Finish Rounded Model no-uniform (round_to=$round_to)"
done
