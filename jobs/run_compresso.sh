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



deepspeed --num_nodes=1 --num_gpus=$NUM_GPUS train.py \
    --deepspeed ds3_offload.json \
    --pruning_type structured_heads+structured_mlp+hidden \
    --target_sparsity 0.3 \
    --sparsity_epsilon 0.005 \
    --model_name_or_path decapoda-research/llama-7b-hf \
    --num_train_epochs 8 \
    --learning_rate 5e-5 \
    --reg_learning_rate 0.1 \
    --lagrangian_warmup_epochs 4 \
    --max_seq_length 1024 \
    --task_name pruning \
    --do_train \
    --do_eval \
    --dataset_name alpaca-gpt4 \
    --eval_dataset_name wikitext \
    --train_file ./data/alpaca_gpt4_data.json \
    --droprate_init 0.01 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --training_objective LM \
    --overwrite_output_dir \
    --output_dir $OUTPUT_DIR/ \
    --cache_dir /dev/shm \
    --use_lora True \
    --lora_rank 8 \
    --lora_train_bias none \
    --lora_alpha 8.0 \
    --lora_param Q.V \
    --lora_layers 32 \
    --gradient_checkpointing=True \
    --logging_first_step \
    --logging_steps 10 \
    --disable_tqdm True \
    --fp16 false \
    --random_init=False



for pruning_ratio in $pruning_ratios; do
    echo "-->>> >>> Pruning Ratio: $pruning_ratio <<< <<<--"

    echo "[START] - Start Pruning Model ($pruning_ratio)"
    python run_compresso.py \
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
