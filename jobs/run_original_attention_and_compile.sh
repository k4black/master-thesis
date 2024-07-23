nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


python -m pip install "flash-attn>=2.6.0.post1,<3.0.0"


#base_model_name='huggyllama/llama-7b'
base_model_name='meta-llama/Llama-2-7b-hf'
#base_model_name='meta-llama/Meta-Llama-3-8B'

attention_type_list="eager flash_attention_2 sdpa"

for attention_type in $attention_type_list; do
    echo "-->>> >>> Attention Type: $attention_type <<< <<<--"

    echo "[START] - Start Original Model (0)"
    python run_original.py \
        --base-model=$base_model_name \
        --attention-type $attention_type \
        --no-pytorch-compile \
        --extra-tags "attention_types" \
        --evaluate-on="perplexity"
    echo "[END] - Finish Original Model (0)"

    echo "[START] - Start Original Model (0)"
    python run_original.py \
        --base-model=$base_model_name \
        --attention-type $attention_type \
        --pytorch-compile \
        --extra-tags "attention_types" \
        --evaluate-on="perplexity"
    echo "[END] - Finish Original Model (0)"
done


