nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate

#base_model_name='huggyllama/llama-7b'
base_model_name='meta-llama/Llama-2-7b-hf'
#base_model_name='meta-llama/Meta-Llama-3-8B'

echo "[START] - Start Original Model (0)"
python run_original.py \
    --base-model=$base_model_name \
    --num-train-epochs 0 \
    --train-batch-size 12 \
    --learning-rate 2e-4 \
    --training-dtype "fp16" \
    --evaluate-on="perplexity+full+bias"
echo "[END] - Finish Original Model (0)"


#echo "[START] - Start Original Model (0)"
#python run_original.py \
#    --base-model=$base_model_name \
#    --num-train-epochs 1 \
#    --train-batch-size 12 \
#    --learning-rate 2e-4 \
#    --training-dtype "fp16" \
#    --evaluate-on="perplexity+full+bias"
#echo "[END] - Finish Original Model (0)"
#
#echo "[START] - Start Original Model (0)"
#python run_original.py \
#    --base-model=$base_model_name \
#    --num-train-epochs 1 \
#    --train-batch-size 12 \
#    --learning-rate 1e-4 \
#    --training-dtype "fp16" \
#    --evaluate-on="perplexity+full+bias"
#echo "[END] - Finish Original Model (0)"
#
#echo "[START] - Start Original Model (0)"
#python run_original.py \
#    --base-model=$base_model_name \
#    --num-train-epochs 1 \
#    --train-batch-size 12 \
#    --learning-rate 5e-5 \
#    --training-dtype "fp16" \
#    --evaluate-on="perplexity+full+bias"
#echo "[END] - Finish Original Model (0)"
