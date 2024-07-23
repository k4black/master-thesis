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

pruning_ratios="0.1 0.2 0.3 0.4 0.5 0.6"
#what_to_prune_list="attn_heads+ffn_neurons attn_heads_uniform+ffn_neurons_uniform"
what_to_prune_list="attn_heads+ffn_neurons"
#what_to_prune_list="attn_heads_uniform+ffn_neurons_uniform"
round_to=256

#
#for what_to_prune in $what_to_prune_list; do
#    echo "-->>> >>> Pruning Components: $what_to_prune <<< <<<--"
#
#    for pruning_ratio in $pruning_ratios; do
#        echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"
#
#        echo "[START] - Start Pruning Model ($pruning_ratio)"
#        python run_our.py \
#            --base-model=$base_model_name \
#            --pruning-components $what_to_prune \
#            --pruning-ratio $pruning_ratio \
#            --batch-size 8 \
#            --round-to $round_to \
#            --how-to-collect "grads" \
#            --how-to-average "fisher_info" \
#            --how-to-overlap "fixed" \
#            --evaluate-on="perplexity+short+bias"
#        echo "[END] - Finish Pruning Model ($pruning_ratio)"
#    done
#done



#pruning_ratios="0.01 0.025 0.05 0.1 0.2"
#what_to_prune_list="hidden_states+attn_heads+ffn_neurons"
#round_to=32
#
#
#for what_to_prune in $what_to_prune_list; do
#    echo "-->>> >>> Pruning Components: $what_to_prune <<< <<<--"
#
#    for pruning_ratio in $pruning_ratios; do
#        echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"
#
#        echo "[START] - Start Pruning Model ($pruning_ratio)"
#        python run_our.py \
#            --base-model=$base_model_name \
#            --pruning-components $what_to_prune \
#            --pruning-ratio $pruning_ratio \
#            --batch-size 8 \
#            --round-to $round_to \
#            --how-to-collect "grads" \
#            --how-to-average "fisher_info" \
#            --how-to-overlap "meta" \
#            --evaluate-on="perplexity+short+bias"
#        echo "[END] - Finish Pruning Model ($pruning_ratio)"
#    done
#done



#python run_our.py \
#    --base-model=$base_model_name \
#    --pruning-components "attn_heads+ffn_neurons" \
#    --pruning-ratio 0.3 \
#    --batch-size 8 \
#    --how-to-collect "grads" \
#    --how-to-average "fisher_info" \
#    --how-to-overlap "fixed" \
#    --num-train-epochs 0 \
#    --num-iterations 1 \
#    --evaluate-on="perplexity+short+bias"

python run_our.py \
    --base-model=$base_model_name \
    --pruning-components "attn_heads+ffn_neurons" \
    --pruning-ratio 0.0 \
    --batch-size 8 \
    --how-to-collect "grads" \
    --how-to-average "fisher_info" \
    --how-to-overlap "fixed" \
    --num-train-epochs 0 \
    --num-iterations 1 \
    --evaluate-on="perplexity+short+bias"

iterations_list="1 4 16"
for num_iterations in $iterations_list; do
    echo "-->>> >>> Number of Iterations: $num_iterations <<< <<<--"

    echo "[START] - Start Pruning Model ($num_iterations)"
    python run_our.py \
        --base-model=$base_model_name \
        --pruning-components "attn_heads+ffn_neurons" \
        --pruning-ratio 0.3 \
        --batch-size 8 \
        --how-to-collect "grads" \
        --how-to-average "fisher_info" \
        --how-to-overlap "fixed" \
        --num-train-epochs 1 \
        --num-iterations $num_iterations \
        --evaluate-on="perplexity+short+bias"
    echo "[END] - Finish Pruning Model ($num_iterations)"
done

python run_our.py \
    --base-model=$base_model_name \
    --pruning-components "attn_heads+ffn_neurons" \
    --pruning-ratio 0 \
    --batch-size 8 \
    --how-to-collect "grads" \
    --how-to-average "fisher_info" \
    --how-to-overlap "fixed" \
    --num-train-epochs 1 \
    --num-iterations 1 \
    --evaluate-on="perplexity+short+bias"