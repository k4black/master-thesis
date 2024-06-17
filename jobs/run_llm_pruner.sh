nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


#base_model_name='TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T'
#base_model_name='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
#prune_ckpt_path='TinyLlama-1.1B-pruned'
base_model_name='huggyllama/llama-7b'
prune_ckpt_path='llama-7b-pruned'

pruning_rations="0.1 0.2 0.3 0.4 0.5"
pruning_rations="0.025 0.05 0.1 0.2 0.3"
#how_to_prune="random l1 taylor"
how_to_prune="taylor"

#echo "[START] - Start Pruning Model (0)"
#python run_llm_pruner.py \
#    --base_model=$base_model_name \
#    --pruning_ratio 0 \
#    --device cuda \
#    --eval_device cuda \
#    --block_wise \
#    --block_mlp_layer_start 4 \
#    --block_mlp_layer_end 21 \
#    --block_attention_layer_start 4 \
#    --block_attention_layer_end 21 \
#    --taylor param_first \
#    --save_ckpt_log_name $prune_ckpt_path \
#    --pruner_type random \
#    --evaluate
#echo "[END] - Finish Pruning Model (0)"

for how_to in $how_to_prune; do
  for pruning_ratio in $pruning_rations; do
      echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

      echo "[START] - Start Pruning Model ($pruning_ratio)"
      python run_llm_pruner.py \
          --base_model=$base_model_name \
          --pruning_ratio $pruning_ratio \
          --device cuda \
          --eval_device cuda \
          --channel_wise \
          --block_mlp_layer_start 4 \
          --block_mlp_layer_end 21 \
          --block_attention_layer_start 4 \
          --block_attention_layer_end 21 \
          --taylor param_first \
          --save_ckpt_log_name $prune_ckpt_path \
          --pruner_type $how_to \
          --evaluate
  #        --block_wise \
  #        --channel_wise \
      #    --save_model
      echo "[END] - Finish Pruning Model ($pruning_ratio)"
  done
done
