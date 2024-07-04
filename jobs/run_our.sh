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
save_as='our-llama-7b-pruned'


echo "[START] - Start Pruning Model (0)"
python run_our.py \
    --base_model=$base_model_name \
    --pruning_components ffn_neurons \
    --pruning_ratio 0.0 \
    --batch_size 10 \
    --evaluate_on="perplexity+full+toxicity"
echo "[END] - Finish Pruning Model (0)"

#what_to_prune="ffn_neurons ffn_neurons_uniform attention_heads attention_heads_uniform"
#what_to_prune="attention_heads ffn_neurons attention_heads+ffn_neurons"
what_to_prune="attention_heads+ffn_neurons"
pruning_ratios="0.05 0.1 0.2 0.3 0.4"
#pruning_ratios="0.2"
#how_to_collect="random weights grads activations"
how_to_collect="grads"
#how_to_overlap="meta relative relative_per_param random fixed"
how_to_overlap="fixed_x2_x05 fixed_x05_x2"

for to_overlap in $how_to_overlap; do
  for to_collect in $how_to_collect; do
    for to_prune in $what_to_prune; do
      for pruning_ratio in $pruning_ratios; do
          echo "-->>> >>> Pruning Ratio: $pruning_ratio to prune: $to_prune <<< <<<--"

          echo "[START] - Start Pruning Model ($pruning_ratio)"
          python run_our.py \
              --base_model=$base_model_name \
              --pruning_components $to_prune \
              --pruning_ratio $pruning_ratio \
              --batch_size 10 \
              --how_to_collect $to_collect \
              --how_to_average fisher_info \
              --how_to_overlap $to_overlap \
              --evaluate
          echo "[END] - Finish Pruning Model ($pruning_ratio)"

    #          --use_cache \
    #          --how_to_collect weights \
    #          --how_to_average mean \
      done
    done
  done
done
