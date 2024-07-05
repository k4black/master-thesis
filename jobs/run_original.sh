nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


base_model_name='meta-llama/Meta-Llama-3-8B'
#base_model_name='huggyllama/llama-7b'

echo "[START] - Start Original Model (0)"
python run_original.py \
    --base-model=$base_model_name \
    --evaluate-on="perplexity+full+bias"
echo "[END] - Finish Original Model (0)"
