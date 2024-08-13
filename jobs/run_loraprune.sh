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

pruning_ratios="0.05 0.1 0.2 0.3 0.4 0.5 0.6"



CUDA_VISIBLE_DEVICES=0 python prune.py \
    --batch_size 128 \
    --micro_batch_size 2 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 1000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj,up_proj, down_proj]' \
    --train_on_inputs \
    --group_by_length \
    --ratio 0.5 \
    --prune_metric 'lora' \
    --prune_freq 10 \


for pruning_ratio in $pruning_ratios; do
    echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

    echo "[START] - Start Pruning Model ($pruning_ratio)"
    python run_loraprune.py \
        --base-model=$base_model_name \
        --pruning-ratio $pruning_ratio \
        --block-wise \
        --pruner-type "taylor" \
        --evaluate-on="perplexity+full+bias"
#        --block-wise \
#        --channel-wise \
    #    --save-model-as TBA
    echo "[END] - Finish Pruning Model ($pruning_ratio)"
done
