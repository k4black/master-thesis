import os


import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_theme(style="whitegrid")

DIRECTORY = 'results/info-7b/'

# Load data from pickle files
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Files
files = {
    'activations_attention_heads': 'activations_attention_heads_info_pickle.pkl',
    'weights_attention_heads': 'weights_attention_heads_info_pickle.pkl',
    'grads_attention_heads': 'grads_attention_heads_info_pickle.pkl',
    # 'grads_attention_heads': 'grads_attention_heads_importance_pickle.pkl',
    'activations_ffn_neurons': 'activations_ffn_neurons_info_pickle.pkl',
    'weights_ffn_neurons': 'weights_ffn_neurons_info_pickle.pkl',
    'grads_ffn_neurons': 'grads_ffn_neurons_info_pickle.pkl',
    # 'grads_ffn_neurons': 'grads_ffn_neurons_importance_pickle.pkl',
}

# Initialize dataframes to store processed data
attention_data = pd.DataFrame()
ffn_neurons_data = pd.DataFrame()

# Process each file
for key, file in files.items():
    data = load_data(DIRECTORY+file)
    avg_data = np.array(data)
    print(key, file, avg_data.shape)

    # if 'activations' in key or 'weights' in key:
    #     avg_data = np.abs(avg_data)

    # average across samples
    avg_data = avg_data.mean(axis=0)

    avg_data = avg_data.flatten()

    # remove outliers (1%)
    lower_percentile = np.percentile(avg_data, 3)
    upper_percentile = np.percentile(avg_data, 97)
    avg_data = avg_data[(avg_data > lower_percentile) & (avg_data < upper_percentile)]

    # Normalize 0 to 1
    avg_data = (avg_data - np.min(avg_data)) / (np.max(avg_data) - np.min(avg_data))

    temp_df = pd.DataFrame({
        'Value (Relative)': avg_data,
        'Type': key.split('_')[0] * avg_data.shape[0],
    })

    if 'attention_heads' in key:
        attention_data = pd.concat([attention_data, temp_df], ignore_index=True)
    elif 'ffn_neurons' in key:
        ffn_neurons_data = pd.concat([ffn_neurons_data, temp_df], ignore_index=True)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
fig.suptitle('Distribution of Relative Values', fontsize=14, fontweight="bold")
plt.tight_layout(h_pad=3, w_pad=2, rect=[0, 0, 0.92, 0.95])
plt.tight_layout(h_pad=3, w_pad=2)

# Attention Heads Plot
sns.histplot(
    data=attention_data,
    x='Value (Relative)',
    hue='Type',
    multiple='dodge',
    stat='density',
    bins=50,
    ax=axes[0],
)
axes[0].set_title('Attention Heads', fontweight="bold")

# FFN Neurons Plot
sns.histplot(
    data=ffn_neurons_data.sample(frac=0.001, random_state=42),
    x='Value (Relative)',
    hue='Type',
    multiple='dodge',
    stat='density',
    bins=50,
    ax=axes[1],
)
axes[1].set_title('FFN Neurons', fontweight="bold")

# Add common legend
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles=handles, labels=labels, loc='center right')
# for ax in axes.flat:
#     if legend := ax.get_legend():
#         legend.remove()

plt.savefig('dist-results.png')
plt.savefig('dist-results.svg')
plt.show()
