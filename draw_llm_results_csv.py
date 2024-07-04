import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")

# Read the CSV file
df = pd.read_csv("master-thesis-7b.csv", index_col=None)
# tiny
# model = 'TinyLlama-1.1B'
# PARAMS_MIN = 0.75
# GFLOPS_MIN = 1400
# PERPLEXITY_MIN = 60
# PIQA_MIN = 0.7
# ARC_EASY_MIN = 0.5
# 7b
model = "LLaMA-7B"
PARAMS_MIN = 5
RELATIVE_PARAMS_MIN = 0.7
GFLOPS_MIN = 10000
PERPLEXITY_MIN = 35
PIQA_MIN = 0.7
BOOLQ_MIN = 0.6
ARC_EASY_MIN = 0.5

MAIN_TITLE = f"{model} pruning \\wo finetuning"  # + " (Heads+Neurons pruning)"


df.loc[df["ratio"] == 0.0, "method"] = "none"
df.loc[df["ratio"] == 0.0, "lib"] = "none"
df.loc[df["ratio"] == 0.0, "how_to_average"] = "none"
df.loc[df["ratio"] == 0.0, "how_to_collect"] = "none"
df.loc[df["ratio"] == 0.0, "how_to_overlap"] = "none"

df["how_to_overlap"] = df["how_to_overlap"].fillna("")

df["pruning_components"] = df["method"].apply(
    lambda x: x.replace("+random", "")
    .replace("+l1", "")
    .replace("+l2", "")
    .replace("+taylor", "")
    .replace("+fisher_info", "")
    .replace("+fixed_x05_x2", "")
    .replace("+fixed_x2_x05", "")
    .replace("+fixed", "")
    .replace("+relative_per_param", "")
    .replace("+relative", "")
    .replace("+relative_per_param", "")
    .replace("+meta", "")
)
df["line_name"] = (
    df["lib"] + "\n" + df["pruning_components"] + "\n" + df["how_to_collect"]
)  # + '\n' + df['how_to_overlap']

df["line_name"] = df["line_name"].replace("none\nnone\nnone", "Original")
original_line = df[df["line_name"] == "Original"].iloc[0]
df = df[df["line_name"] != "Original"]
df = df._append(original_line, ignore_index=True)

# drop rows with not available results
df = df[df["params"].notna()]

# drop rows where params are == Original (keep Original row)
original_params = df[df["line_name"] == "Original"]["params"].values[0]
df = df[(df["params"] != original_params) | (df["line_name"] == "Original")]

# MANUAL filtering
# keep only lines with names
keep_line_names_and_rename = {
    "original": "Original",
    "our\nattn-heads+ffn-neurons\ngrads": "OUR\nHeads+Neurons",
    "our\nattn-heads\ngrads": "OUR\nHeads",
    "our\nffn-neurons\ngrads": "OUR\nNeurons",
    "llm-pruner\nattn-heads+ffn-neurons\ntaylor": "LLM-Pruner\nHeads+Neurons",
    "llm-pruner\nattn-heads+ffn-neurons\nrandom": "LLM-Pruner\nRandom",
    "llm-pruner\nattn-heads+ffn-neurons\nl1": "LLM-Pruner\nWeights L1",
    "llm-pruner\nhidden-state\ntaylor": "LLM-Pruner\nHidden-State",
    # 'our\nattn-heads+ffn-neurons\ngrads\nfixed': "OUR\nFixed_x1_x1",
    # 'our\nattn-heads+ffn-neurons\ngrads\nfixed_x2_x05': "OUR\nFixed_x2_x05",
    # 'our\nattn-heads+ffn-neurons\ngrads\nfixed_x05_x2': "OUR\nFixed_x05_x2",
    # 'our\nattn-heads+ffn-neurons\ngrads\nrelative': "OUR\nRelative",
    # 'our\nattn-heads+ffn-neurons\ngrads\nrelative_per_param': "OUR\nRelative_per_param",
    # 'our\nattn-heads+ffn-neurons\ngrads\nmeta': "OUR\nMeta",
    # 'llm-pruner\nattn-heads+ffn-neurons\ntaylor\n': "LLM-Pruner",
}
print(df["line_name"].unique())
df = df[df["line_name"].str.lower().isin(keep_line_names_and_rename.keys())]
df["line_name"] = df["line_name"].str.lower().map(keep_line_names_and_rename)
print(df["line_name"].unique())

# sort by line_name with first being Original, then starting with 'our', then others
df = df.sort_values(
    by=["line_name"],
    key=lambda x: -2 * x.str.contains("Original").astype(int) - x.str.lower().str.contains("our").astype(int),
    ignore_index=True,
)
unique_lines = list(df["line_name"].unique())
print(unique_lines)

# add converted columns
df["GFLOPs"] = df["flops"] / 1e9
df["TFLOPs"] = df["flops"] / 1e12
df["GMACs"] = df["macs"] / 1e9
df["#Parameters (1e9)"] = df["params"] / 1e9
df["Relative #Parameters"] = df["params"] / original_line["params"]
df["Samples/s"] = df["samples_per_sec"]
df["speed"] = df["samples_per_sec"]
df["ms/sample"] = 1 / df["samples_per_sec"] * 1e3
df["Ratio (\%)"] = (df["ratio"] * 100).round(0).astype(int)

original_line = df[df["line_name"].str.lower().str.contains("original")].iloc[0]

rows_params = ["Relative #Parameters"]
columns_metrics = ["wikitext", "piqa", "boolq", "arc_easy"]
columns_metrics_names = ["perplexity", "accuracy", "accuracy", "accuracy"]


# print latex table (group by Lib than Method)
df_for_latex = df[
    ["lib", "method", "Ratio (\%)", "#Parameters (1e9)", "GFLOPs", "wikitext", "piqa", "boolq", "arc_easy"]
]
df_for_latex = df_for_latex.sort_values(by=["lib", "method", "Ratio (\%)"], ignore_index=True)
print(df_for_latex.to_latex(index=False, float_format="%.2f"))

# Use sns to create line plots - subplot - columns for each metric, rows for each param
fig, axs = plt.subplots(len(rows_params), len(columns_metrics), figsize=(15, 3.5 * len(rows_params)))
fig.suptitle(MAIN_TITLE, fontsize=14, fontweight="bold")
# add space to the right for legend
fig.tight_layout(h_pad=3, w_pad=2, rect=[0, 0, 0.89, 1])
for i, row_param in enumerate(rows_params):
    for j, (column_metric, column_metric_name) in enumerate(zip(columns_metrics, columns_metrics_names)):
        try:
            ax = axs[i, j]
        except IndexError:
            ax = axs[j]

        # add line_name=none/none as grey dashed x/y axis
        ax.axhline(original_line[column_metric], color="grey", linestyle="--", linewidth=1, alpha=0.75)
        ax.axvline(original_line[row_param], color="grey", linestyle="--", linewidth=1, alpha=0.75)

        # draw plot
        # dashed line for Our method
        sns.lineplot(
            data=df,
            x=row_param,
            y=column_metric,
            hue="line_name",
            style="line_name",
            dashes={k: (4, 2) if "our" not in k.lower() and "original" not in k.lower() else "" for k in unique_lines},
            markers=True,
            ax=ax,
        )
        ax.set_title(f"{column_metric} ({column_metric_name})", fontweight="bold")
        # ax.set_ylabel(column_metric)
        # ax.set_xlabel('epoch')
        if column_metric_name == "perplexity":
            ax.set_ylim(original_line[column_metric] - 3, PERPLEXITY_MIN)
            ax.invert_yaxis()
        if column_metric == "piqa":
            ax.set_ylim(PIQA_MIN, None)
        if column_metric == "boolq":
            ax.set_ylim(BOOLQ_MIN, None)
        if column_metric == "arc_easy":
            ax.set_ylim(ARC_EASY_MIN, None)
        if row_param == "#Parameters (1e9)":
            ax.set_xlim(PARAMS_MIN, None)
        if row_param == "Relative #Parameters":
            ax.set_xlim(RELATIVE_PARAMS_MIN, None)
        if row_param == "GFLOPs":
            ax.set_xlim(GFLOPS_MIN, None)

try:
    handles, labels = axs[0, 0].get_legend_handles_labels()
except IndexError:
    handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles=handles, labels=labels, loc="center right")
for ax in axs.flat:
    if legend := ax.get_legend():
        legend.remove()

plt.savefig("llm-results.png")
plt.savefig("llm-results.svg")
plt.show()
