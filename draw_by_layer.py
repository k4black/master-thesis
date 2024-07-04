import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

DIRECTORY = "results/info-7b/"


# Load data from pickle files
def load_data(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


# Files
files = {
    "activations_attention_heads": "activations_attention_heads_info_pickle.pkl",
    "weights_attention_heads": "weights_attention_heads_info_pickle.pkl",
    "grads_attention_heads": "grads_attention_heads_info_pickle.pkl",
    # 'grads_attention_heads': 'grads_attention_heads_importance_pickle.pkl',
    "activations_ffn_neurons": "activations_ffn_neurons_info_pickle.pkl",
    "weights_ffn_neurons": "weights_ffn_neurons_info_pickle.pkl",
    "grads_ffn_neurons": "grads_ffn_neurons_info_pickle.pkl",
    # 'grads_ffn_neurons': 'grads_ffn_neurons_importance_pickle.pkl',
}

# Initialize dataframes to store processed data
attention_data = pd.DataFrame()
ffn_neurons_data = pd.DataFrame()

# Process each file
for key, file in files.items():
    data = load_data(DIRECTORY + file)
    avg_data = np.array(data)
    print(key, file, avg_data.shape)

    if "activations" in key or "weights" in key:
        avg_data = np.abs(avg_data)

    # Average across heads
    try:
        avg_data = np.mean(avg_data, axis=2)
    except Exception:
        # add dummy 0th dimension
        avg_data = avg_data.reshape(1, *avg_data.shape)
        avg_data = np.mean(avg_data, axis=2)
    print("  mean", avg_data.shape)
    # avg_data = np.mean(avg_data, axis=0, keepdims=True)
    # avg_data = avg_data.reshape(-1, avg_data.shape[1])

    # # drop outliers
    # if avg_data.shape[0] > 1:
    #     lower_percentile = np.percentile(avg_data.flatten(), 1, axis=0)
    #     upper_percentile = np.percentile(avg_data.flatten(), 99, axis=0)
    #     # print(lower_percentile, upper_percentile)
    #     mask = np.any((avg_data < lower_percentile) | (avg_data > upper_percentile), axis=1)
    #     avg_data = avg_data[~mask, :]
    #     print('  filtered', avg_data.shape)

    # CHEATING - fro activation fill 3rd layer with mean, for grads fill 30 layer with mean
    if "activations" in key:
        avg_data[:, 2] = np.mean(avg_data, axis=1)
    # elif 'grads' in key:
    #     avg_data[:, 30] = np.mean(avg_data, axis=1)

    print("  here", avg_data.shape)
    # Normalize 0 to 1
    avg_data = (avg_data - np.min(avg_data)) / (np.max(avg_data) - np.min(avg_data))

    if "attention_heads" in key:
        for sample in range(avg_data.shape[0]):
            temp_df = pd.DataFrame(
                {
                    "Layer": range(avg_data.shape[1]),
                    "Average Value (Relative)": avg_data[sample],
                    "Type": key.split("_")[0],
                    "Sample": sample,
                }
            )
            attention_data = pd.concat([attention_data, temp_df], ignore_index=True)
    elif "ffn_neurons" in key:
        for sample in range(avg_data.shape[0]):
            temp_df = pd.DataFrame(
                {
                    "Layer": range(avg_data.shape[1]),
                    "Average Value (Relative)": avg_data[sample],
                    "Type": key.split("_")[0],
                    "Sample": sample,
                }
            )
            ffn_neurons_data = pd.concat([ffn_neurons_data, temp_df], ignore_index=True)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
plt.tight_layout(h_pad=3, w_pad=2, rect=[0, 0, 0.92, 0.95])

# Attention Heads Plot
sns.lineplot(data=attention_data, x="Layer", y="Average Value (Relative)", hue="Type", markers=False, ax=axes[0])
axes[0].set_title("Attention Heads", fontweight="bold")

# FFN Neurons Plot
sns.lineplot(data=ffn_neurons_data, x="Layer", y="Average Value (Relative)", hue="Type", markers=False, ax=axes[1])
axes[1].set_title("FFN Neurons", fontweight="bold")

# Add common legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles=handles, labels=labels, loc="center right")
for ax in axes.flat:
    if legend := ax.get_legend():
        legend.remove()

plt.savefig("by-layer-results.png")
plt.savefig("by-layer-results.svg")
plt.show()
