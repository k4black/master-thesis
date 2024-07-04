nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


base_model_name='huggyllama/llama-7b'

echo "[START] - Start Original Model (0)"
python run_original.py \
    --base_model=$base_model_name \
    --evaluate_on="perplexity+full+toxicity"
echo "[END] - Finish Original Model (0)"
