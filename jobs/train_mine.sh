nvidia-smi || true
# source .env
export HF_HOME=/netscratch/kchernyshev/.cache/huggingface
export $(cat .env | xargs)
# activate the virtual environment
. .venv/bin/activate



prune_components="do_prune_ffn_neurons do_prune_attention_heads"


#for prune_component in $prune_components; do
#  echo "-------------------"
#  echo "Running with prune_component: $prune_component"
#  echo "-------------------"
#  python main.py --$prune_component
#done

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#do_prune_ffn_neurons
#   remain_percentage  metric  elapsed_time  params_num params_num_str  relative_metric  relative_elapsed_time  relative_params_num
#0                1.0  0.8468          6.61   109484547         109.5M             1.00                   1.00                 1.00
#1                0.9  0.8449          8.26   103822239         103.8M             1.00                   1.25                 0.95
#2                0.8  0.8397          6.96    98159931          98.2M             0.99                   1.05                 0.90
#3                0.7  0.8344          7.70    92497623          92.5M             0.99                   1.16                 0.84
#4                0.6  0.8295          6.59    86835315          86.8M             0.98                   1.00                 0.79
#5                0.5  0.8063          5.85    81154563          81.2M             0.95                   0.89                 0.74
#6                0.4  0.7699          6.69    75492255          75.5M             0.91                   1.01                 0.69
#7                0.3  0.7164          5.87    69829947          69.8M             0.85                   0.89                 0.64
#8                0.2  0.6581          5.89    64167639          64.2M             0.78                   0.89                 0.59
#9                0.1  0.4558          5.37    58505331          58.5M             0.54                   0.81                 0.53
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#do_prune_attention_heads
#   remain_percentage  metric  elapsed_time  params_num params_num_str  relative_metric  relative_elapsed_time  relative_params_num
#0                1.0  0.8468          6.84   109484547         109.5M             1.00                   1.00                 1.00
#1                0.9  0.8453          6.50   107122947         107.1M             1.00                   0.95                 0.98
#2                0.8  0.8424          6.33   104761347         104.8M             0.99                   0.93                 0.96
#3                0.7  0.8401          6.18   102399747         102.4M             0.99                   0.90                 0.94
#4                0.6  0.8388          5.94   100038147         100.0M             0.99                   0.87                 0.91
#5                0.5  0.8244          5.54    95314947          95.3M             0.97                   0.81                 0.87
#6                0.4  0.8040          5.38    92953347          93.0M             0.95                   0.79                 0.85
#7                0.3  0.7526          5.19    90591747          90.6M             0.89                   0.76                 0.83
#8                0.2  0.6038          5.05    88230147          88.2M             0.71                   0.74                 0.81
#9                0.1  0.5171          4.88    85868547          85.9M             0.61                   0.71                 0.78



for prune_component in $prune_components; do
  echo "-------------------"
  echo "Running with prune_component: $prune_component --do_lora_finetuning"
  echo "-------------------"
  python main.py --$prune_component --do_lora_finetuning
done


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# do_prune_ffn_neurons --do_lora_finetuning
#   remain_percentage  metric  elapsed_time  params_num params_num_str  relative_metric  relative_elapsed_time  relative_params_num
#0                1.0  0.8468          8.23   109779459         109.8M             1.00                   1.00                 1.00
#1                0.9  0.8440         10.01   104117151         104.1M             1.00                   1.22                 0.95
#2                0.8  0.8407          8.69    98454843          98.5M             0.99                   1.06                 0.90
#3                0.7  0.8365          9.34    92792535          92.8M             0.99                   1.13                 0.85
#4                0.6  0.8269          8.24    87130227          87.1M             0.98                   1.00                 0.79
#5                0.5  0.8043          7.43    81449475          81.4M             0.95                   0.90                 0.74
#6                0.4  0.7675          8.26    75787167          75.8M             0.91                   1.00                 0.69
#7                0.3  0.7130          7.43    70124859          70.1M             0.84                   0.90                 0.64
#8                0.2  0.6393          7.41    64462551          64.5M             0.75                   0.90                 0.59
#9                0.1  0.4673          6.86    58800243          58.8M             0.55                   0.83                 0.54
#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# do_prune_attention_heads --do_lora_finetuning
#   remain_percentage  metric  elapsed_time  params_num params_num_str  relative_metric  relative_elapsed_time  relative_params_num
#0                1.0  0.8468          8.19   109779459         109.8M             1.00                   1.00                 1.00
#1                0.9  0.8446          7.96   107405571         107.4M             1.00                   0.97                 0.98
#2                0.8  0.8442          7.70   105031683         105.0M             1.00                   0.94                 0.96
#3                0.7  0.8418          7.45   102657795         102.7M             0.99                   0.91                 0.94
#4                0.6  0.8358          7.11   100283907         100.3M             0.99                   0.87                 0.91
#5                0.5  0.8159          6.58    95536131          95.5M             0.96                   0.80                 0.87
#6                0.4  0.8075          6.39    93162243          93.2M             0.95                   0.78                 0.85
#7                0.3  0.7855          6.13    90788355          90.8M             0.93                   0.75                 0.83
#8                0.2  0.6830          5.99    88414467          88.4M             0.81                   0.73                 0.81
#9                0.1  0.5633          5.83    86040579          86.0M             0.67                   0.71                 0.78


for prune_component in $prune_components; do
  echo "-------------------"
  echo "Running with prune_component: $prune_component --do_full_finetuning"
  echo "-------------------"
  python main.py --$prune_component --do_full_finetuning
done


#
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# do_prune_ffn_neurons --do_full_finetuning
#   remain_percentage  metric  elapsed_time  params_num params_num_str  relative_metric  relative_elapsed_time  relative_params_num
#0                1.0  0.8370          7.12   109484547         109.5M             1.00                   1.00                 1.00
#1                0.9  0.8374          8.98   103822239         103.8M             1.00                   1.26                 0.95
#2                0.8  0.8320          7.58    98159931          98.2M             0.99                   1.06                 0.90
#3                0.7  0.8230          8.29    92497623          92.5M             0.98                   1.16                 0.84
#4                0.6  0.8064          7.09    86835315          86.8M             0.96                   1.00                 0.79
#5                0.5  0.7871          6.15    81154563          81.2M             0.94                   0.86                 0.74
#6                0.4  0.7507          7.07    75492255          75.5M             0.90                   0.99                 0.69
#7                0.3  0.7151          6.19    69829947          69.8M             0.85                   0.87                 0.64
#8                0.2  0.5915          6.18    64167639          64.2M             0.71                   0.87                 0.59
#9                0.1  0.5438          5.63    58505331          58.5M             0.65                   0.79                 0.53
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# do_prune_attention_heads --do_full_finetuning
   remain_percentage  metric  elapsed_time  params_num params_num_str  relative_metric  relative_elapsed_time  relative_params_num
#0                1.0  0.8370          7.00   109484547         109.5M             1.00                   1.00                 1.00
#1                0.9  0.8388          6.87   107122947         107.1M             1.00                   0.98                 0.98
#2                0.8  0.8382          6.69   104761347         104.8M             1.00                   0.96                 0.96
#3                0.7  0.8368          6.53   102399747         102.4M             1.00                   0.93                 0.94
#4                0.6  0.8307          6.29   100038147         100.0M             0.99                   0.90                 0.91
#5                0.5  0.8123          5.87    95314947          95.3M             0.97                   0.84                 0.87
#6                0.4  0.7685          5.70    92953347          93.0M             0.92                   0.81                 0.85
#7                0.3  0.6852          5.42    90591747          90.6M             0.82                   0.77                 0.83
#8                0.2  0.5238          5.34    88230147          88.2M             0.63                   0.76                 0.81
#9                0.1  0.3297          5.17    85868547          85.9M             0.39                   0.74                 0.78
