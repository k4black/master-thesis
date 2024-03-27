nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate

#    --tasks wikitext,openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \

#lm_eval --model hf \
#    --model_args pretrained=bert-base-uncased,dtype=auto \
#    --tasks wikitext,piqa,boolq \
#    --device cuda:0 \
#    --seed 42 \
#    --cache_requests true \
#    --batch_size 32

#lm_eval --model hf \
#    --model_args pretrained=openai-community/gpt2,dtype=auto \
#    --tasks wikitext,piqa,boolq \
#    --device cuda:0 \
#    --seed 42 \
#    --cache_requests true \
#    --batch_size auto

#accelerate launch --mixed_precision fp16 \
#    -m lm_eval \
#      --model hf \
#      --model_args pretrained=openai-community/gpt2-xl,dtype=auto \
#      --tasks wikitext,piqa,boolq \
#      --device cuda:0 \
#      --seed 42 \
#      --cache_requests true \
#      --batch_size auto

#lm_eval \
#    --model vllm \
#    --model_args pretrained=openai-community/gpt2,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
#    --tasks wikitext,piqa,boolq \
#    --seed 42 \
#    --cache_requests true \
#    --batch_size auto:4
##    --device cuda:0 \

#lm_eval \
#    --model hf \
#    --model_args pretrained=openai-community/gpt2-xl,parallelize=True,dtype=auto \
#    --tasks wikitext,piqa,boolq \
#    --seed 42 \
#    --cache_requests true \
#    --batch_size auto:4
##    --device cuda:0 \

#
#lm_eval \
#    --model vllm \
#    --model_args pretrained=openai-community/gpt2-xl,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.9,data_parallel_size=1 \
#    --tasks wikitext,piqa,boolq \
#    --seed 42 \
#    --cache_requests true \
#    --batch_size auto:4
##    --device cuda:0 \



# huggyllama/llama-7b
# meta-llama/Llama-2-7b-hf
# lmsys/vicuna-7b-v1.5
# google/gemma-7b
# TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T
lm_eval \
    --model vllm \
    --model_args pretrained=TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T,tensor_parallel_size=1,dtype=float16,gpu_memory_utilization=0.9,data_parallel_size=1 \
    --tasks wikitext,piqa,boolq \
    --cache_requests true \
    --batch_size auto:4
#    --device cuda:0 \

#python hf_prune.py --base_model=huggyllama/llama-7b --pruning_ratio 0.25 --device cpu --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model
