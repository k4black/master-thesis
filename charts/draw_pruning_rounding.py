from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


sns.set_theme(style="whitegrid", color_codes=True)
DATA_CSV = Path(__file__).parent / "masters-thesis-pruning-rounding.csv"
df = pd.read_csv(DATA_CSV, index_col=None)

# set chart params
MODEL_NAME = "LLaMA-2-7b"
DEVICE_NAME = "A100-40GB"
PRUNING_RATIO = 0.3

# set columns
df["round_to"] = df["how_to_average"].apply(lambda x: int(x.split("=")[-1]))
df["is_uniform"] = df["how_to_overlap"].apply(
    lambda x: "Uniform Pruning" if "is_uniform=True" in x else "Non-Uniform Pruning"
)

# filter dataset
df = df[df["round_to"] != 127]

# get original
original_row = df[df["pruning_ratio"] == 0.0].iloc[0]
df = df[df["pruning_ratio"] != 0.0]

# plot
fig = plt.figure(figsize=(5, 4), dpi=300)
fig.tight_layout()
ax = plt.gca()

# plot original line
line = plt.axhline(original_row["inference_time_average"], linestyle="--", label="Original", color="red")

# plot by rounding on x-axis and inference time on y-axis; hue by uniformity
ax = sns.barplot(
    data=df,
    x="round_to",
    y="inference_time_average",
    hue="is_uniform",
    errorbar=None,
    dodge=True,
    # width=0.8,
    palette="Set2",
    ax=ax,
)

# set labels
plt.title(f"{PRUNING_RATIO*100:.0f}% FFN pruned {MODEL_NAME} ({DEVICE_NAME})", fontsize=13, fontweight="bold")
plt.xlabel("Rounding to X-neurons to Prune")
plt.ylabel("Inference Time (s)")

# Set a different hatch for each group of bars (original is not included)
hatches = ["", "/"]
for container, hatch, handle in zip(ax.containers, hatches, ax.get_legend().legend_handles[1:]):
    print(container, hatch, handle)
    handle.set_hatch(hatch)
    for rectangle in container:
        rectangle.set_hatch(hatch)

# change name in legend
ax.legend(
    handles=ax.get_legend().legend_handles,
    title="",
    labels=["Original", "Non-Uniform", "Uniform"],
    loc="lower right",
).get_frame().set_alpha(0.95)


# save and show
plt.savefig(Path(__file__).parent / "inference_time_rounding.svg")
plt.savefig(Path(__file__).parent / "inference_time_rounding.png", dpi=600)
plt.show()
