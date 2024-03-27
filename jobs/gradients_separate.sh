nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate



#how_to_average_list="mean fisher_info entropy"
how_to_average_list="fisher_info"
#prune_components_list="do_prune_attention_heads do_prune_attention_heads_uniform do_prune_attention_layers do_prune_ffn_neurons do_prune_ffn_neurons_uniform do_prune_ffn_layers"
prune_components_list="do_prune_attention_heads_uniform do_prune_attention_layers do_prune_ffn_neurons_uniform do_prune_ffn_layers"


for how_to_average in $how_to_average_list; do
  for prune_component in $prune_components_list; do
    echo "-------------------"
    echo "Running with prune_component: $prune_component and how_to_average: $how_to_average"
    echo "-------------------"
    python main.py --num_samples=16384 --num_valid_samples=100000 --$prune_component --how_to_average=$how_to_average --how_to_collect=grads --do_full_finetuning
  done
done

for how_to_average in $how_to_average_list; do
  for prune_component in $prune_components_list; do
    echo "-------------------"
    echo "Running with prune_component: $prune_component and how_to_average: $how_to_average"
    echo "-------------------"
    python main.py --num_samples=16384 --num_valid_samples=100000 --$prune_component --how_to_average=$how_to_average --how_to_collect=grads --do_lora_finetuning
  done
done

#for prune_component in $prune_components_list; do
#  echo "-------------------"
#  echo "Running with prune_component: $prune_component and how_to_average: random"
#  echo "-------------------"
#  python main.py --num_samples=16384 --num_valid_samples=100000 --$prune_component --how_to_collect=random --do_full_finetuning
#done
