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

pruning_ratios="0.2"
what_to_prune_list="attn_heads+ffn_neurons"
round_to=256


for what_to_prune in $what_to_prune_list; do
    echo "-->>> >>> Pruning Components: $what_to_prune <<< <<<--"

    for pruning_ratio in $pruning_ratios; do
        echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

        echo "[START] - Start Pruning Model ($pruning_ratio)"
        python run_our.py \
            --base-model=$base_model_name \
            --pruning-components $what_to_prune \
            --pruning-ratio $pruning_ratio \
            --batch-size 4 \
            --round-to $round_to \
            --how-to-collect "grads" \
            --how-to-average "fisher_info" \
            --how-to-overlap "fixed" \
            --num-train-epochs 0 \
            --num-iterations 4 \
            --training-dtype "fp16" \
            --pruning-dataset "alpaca-gpt4" \
            --evaluate-on="perplexity+short" \
            --extra-tags "our_trainer_error"
        echo "[END] - Finish Pruning Model ($pruning_ratio)"

        echo "[START] - Start Pruning Model ($pruning_ratio)"
        python run_our.py \
            --base-model=$base_model_name \
            --pruning-components $what_to_prune \
            --pruning-ratio $pruning_ratio \
            --batch-size 4 \
            --round-to $round_to \
            --how-to-collect "grads" \
            --how-to-average "fisher_info" \
            --how-to-overlap "fixed" \
            --num-train-epochs 0 \
            --num-iterations 16 \
            --training-dtype "fp16" \
            --pruning-dataset "alpaca-gpt4" \
            --evaluate-on="perplexity+short" \
            --extra-tags "our_trainer_error"
        echo "[END] - Finish Pruning Model ($pruning_ratio)"

        echo "[START] - Start Pruning Model ($pruning_ratio)"
        python run_our.py \
            --base-model=$base_model_name \
            --pruning-components $what_to_prune \
            --pruning-ratio $pruning_ratio \
            --batch-size 4 \
            --round-to $round_to \
            --how-to-collect "grads" \
            --how-to-average "fisher_info" \
            --how-to-overlap "fixed" \
            --num-train-epochs 0 \
            --num-iterations 4 \
            --training-dtype "fp16" \
            --pruning-dataset "c4" \
            --evaluate-on="perplexity+short" \
            --extra-tags "our_trainer_error"
        echo "[END] - Finish Pruning Model ($pruning_ratio)"

        echo "[START] - Start Pruning Model ($pruning_ratio)"
        python run_our.py \
            --base-model=$base_model_name \
            --pruning-components $what_to_prune \
            --pruning-ratio $pruning_ratio \
            --batch-size 4 \
            --round-to $round_to \
            --how-to-collect "grads" \
            --how-to-average "fisher_info" \
            --how-to-overlap "fixed" \
            --num-train-epochs 0 \
            --num-iterations 16 \
            --training-dtype "fp16" \
            --pruning-dataset "c4" \
            --evaluate-on="perplexity+short" \
            --extra-tags "our_trainer_error"
        echo "[END] - Finish Pruning Model ($pruning_ratio)"
    done
done

