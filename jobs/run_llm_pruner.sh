nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


base_model_name='huggyllama/llama-7b'
save_as='llm-pruner-llama-7b-pruned'

pruning_rations="0.1 0.2 0.3 0.4 0.5"
pruning_rations="0.025 0.05 0.1 0.2 0.3"
#how_to_prune="random l1 taylor"
how_to_prune="taylor"


for how_to in $how_to_prune; do
  for pruning_ratio in $pruning_rations; do
      echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

      echo "[START] - Start Pruning Model ($pruning_ratio)"
      python run_llm_pruner.py \
          --base_model=$base_model_name \
          --pruning_ratio $pruning_ratio \
          --channel_wise \
          --save_ckpt_log_name $prune_ckpt_path \
          --pruner_type $how_to \
          --evaluate_on="perplexity+full+toxicity"
  #        --block_wise \
  #        --channel_wise \
      #    --save_model
      echo "[END] - Finish Pruning Model ($pruning_ratio)"
  done
done
