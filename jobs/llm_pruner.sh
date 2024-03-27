nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate


base_model_name='TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T'
prune_ckpt_path='TinyLlama-1.1B-pruned'
#base_model_name='huggyllama/llama-7b'
#prune_ckpt_path='llama-7b-pruned'

pruning_rations="0.4 0.5"
for pruning_ratio in $pruning_rations; do
    echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

    echo "[START] - Start Pruning Model ($pruning_ratio)"
    python llm_pruner_prune.py \
        --base_model=$base_model_name \
        --pruning_ratio $pruning_ratio \
        --device cuda \
        --eval_device cuda \
        --block_wise \
        --block_mlp_layer_start 4 \
        --block_mlp_layer_end 21 \
        --block_attention_layer_start 4 \
        --block_attention_layer_end 21 \
        --pruner_type random \
        --taylor param_first \
        --save_ckpt_log_name $prune_ckpt_path \
        --evaluate
    #    --save_model
#        --pruner_type taylor \
    echo "[END] - Finish Pruning Model ($pruning_ratio)"

    #echo "[START] - Evaluate Pruned Model ($pruning_ratio)"
    #python llm_pruner_evaluate.py \
    #    --base_model $base_model_name \
    #    --checkpoint results/llm_pruner/$prune_ckpt_path/pytorch_model.bin \
    #    --dtype float16 \
    #    --device cuda \
    #    --tasks wikitext,piqa,boolq \
    #    --batch_size auto:4
    #echo "[END] - Finish Evaluation of Pruned Model ($pruning_ratio)"
done
