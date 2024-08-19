nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


base_model_name='meta-llama/Llama-2-13b-hf'

pruning_ratio_list="0.5"
#pruning_ratio_list="0.1 0.25"
#iterations_list="1 4 16"
iterations_list="16"
for pruning_ratio in $pruning_ratio_list; do
  for num_iterations in $iterations_list; do
      echo "-->>> >>> Number of Iterations: $num_iterations <<< <<<--"

      echo "[START] - Start Pruning Model ($num_iterations)"
      python run_our.py \
          --base-model=$base_model_name \
          --pruning-components "attn_heads+ffn_neurons" \
          --how-to-overlap "fixed" \
          --pruning-ratio $pruning_ratio \
          --batch-size 2 \
          --round-to 256 \
          --num-train-epochs 0 \
          --training-dtype "fp16" \
          --num-iterations $num_iterations \
          --evaluate-on="perplexity+full+bias" \
          --extra-tags "new+13b"
      echo "[END] - Finish Pruning Model ($num_iterations)"

      echo "[START] - Start Pruning Model ($num_iterations)"
      python run_our.py \
          --base-model=$base_model_name \
          --pruning-components "attn_heads+ffn_neurons" \
          --how-to-overlap "fixed" \
          --pruning-ratio $pruning_ratio \
          --batch-size 2 \
          --round-to 256 \
          --no-prune-before-training \
          --num-train-epochs 1 \
          --train-batch-size 2 \
          --learning-rate 2e-4 \
          --training-dtype "fp16" \
          --num-iterations $num_iterations \
          --evaluate-on="perplexity+full+bias" \
          --extra-tags "new+13b"
      echo "[END] - Finish Pruning Model ($num_iterations)"

#      echo "[START] - Start Pruning Model ($num_iterations)"
#      python run_our.py \
#          --base-model=$base_model_name \
#          --pruning-components "attn_heads+ffn_neurons" \
#          --how-to-overlap "meta" \
#          --pruning-ratio $pruning_ratio \
#          --batch-size 8 \
#          --round-to 128 \
#          --no-prune-before-training \
#          --num-train-epochs 1 \
#          --num-prune-epochs 0.5 \
#          --train-batch-size 12 \
#          --learning-rate 2e-4 \
#          --training-dtype "fp16" \
#          --num-iterations $num_iterations \
#          --evaluate-on="perplexity+full+bias"
#      echo "[END] - Finish Pruning Model ($num_iterations)"
  done
done
