from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", color_codes=True)
DATA_CSV = Path(__file__).parent / "masters-thesis-compare-2024-jun-15.csv"
df = pd.read_csv(DATA_CSV, index_col=None)

# set chart params
MODEL_NAME = "LLaMA-2-7b"
DEVICE_NAME = "A100-80GB"

ORIGINAL_OPACITY = 0.5

# ROW_LIMITS = (49, 101)
ROW_LIMITS = (50, 100)
# ROW_LIMITS = (70, 100)
METRICS_LIMITS = {
    "wikitext": (35, 5),
    "boolq": (0.6, None),
    "arc_easy": (0.6, None),
    "piqa": (0.7, None),
    "winogrande": (0.6, None),
    "openbookqa": (0.23, None),
    "hellaswag": (0.4, None),
    "arc_challenge": (0.3, None),
    "short_average": (0.65, None),
    "full_average": (0.5, None),
    # 'truthfulqa_mc1': (0.5, None),
    "crows_pairs_english": (0.6, None),
    "inference_time_average": (100, None),
}


# set name
def row_to_name(row):
    if row["lib"] == "original":
        return "Original"

    elif row["lib"] == "wanda":
        if "weights-" in row["pruning_components"]:
            return f'WANDA ({row["pruning_components"].split("-")[-1]})'
        else:
            return "WANDA (sparse)"

    elif row["lib"] == "our":
        name = "Our"
        if row["calibration_how_to_overlap"] == "meta":
            name += ":Meta"
        name += "\n"
        name += {
            "attn_heads+ffn_neurons": "(Heads+FFN)",
            "attn_heads": "(Heads)",
            "ffn_neurons": "(FFN)",
            "hidden_states": "(Hidden)",
            "hidden_states+attn_heads+ffn_neurons": "(Hidden+Heads+FFN)",
            "attn_heads_uniform+ffn_neurons_uniform": "(Un. Heads+FFN)",
        }[row["pruning_components"]]
        return name

    elif row["lib"] == "llm-pruner":
        return "LLM-Pruner"

    else:
        return row["lib"]


df["name"] = df.apply(row_to_name, axis=1)
df = df.sort_values("name")

# get names
ROW = "percent_nonzero_left"
# MODELS = ['Original', 'LLM-Pruner', 'WANDA (2:4)', 'WANDA (4:8)', 'Our\n(Heads+FFN)', 'Our\n(Hidden)', 'Our\n(Hidden+Heads+FFN)', 'Our:Meta\n(Hidden+Heads+FFN)']
MODELS_ALL = [
    "Original",
    "LLM-Pruner",
    "WANDA (2:4)",
    "WANDA (4:8)",
    "Our\n(Heads+FFN)",
    "Our:Meta\n(Hidden+Heads+FFN)",
]
MODELS_OUR = [
    "Original",
    "Our\n(Heads+FFN)",
    "Our\n(Un. Heads+FFN)",
    "Our\n(Hidden)",
    "Our\n(Hidden+Heads+FFN)",
    "Our:Meta\n(Hidden+Heads+FFN)",
]
# METRICS = ['wikitext', 'boolq', 'arc_easy', 'piqa', 'winogrande', 'openbookqa', 'hellaswag', 'arc_challenge', 'short_average', 'full_average', 'truthfulqa_mc1', 'crows_pairs_english', 'inference_time_average']
METRICS = [
    "wikitext",
    "boolq",
    "arc_easy",
    "piqa",
    "short_average",
    "truthfulqa_mc1",
    "crows_pairs_english",
    "inference_time_average",
]
PRETTY_NAMES_METRICS = {
    "percent_nonzero_left": ("%Parameters", ""),
    "inference_time_average": ("Inference Time", "s"),
    "wikitext": ("WikiText", "ppl."),
    "boolq": ("BoolQ", "acc."),
    "arc_easy": ("ARC-Easy", "acc."),
    "piqa": ("PIQA", "acc."),
    "winogrande": ("Winogrande", "acc."),
    "openbookqa": ("OpenBookQA", "acc."),
    "hellaswag": ("HellaSwag", "acc."),
    "arc_challenge": ("ARC-Challenge", "acc."),
    "truthfulqa_mc1": ("TruthfulQA-MC1", "acc."),
    "crows_pairs_english": ("Crows-Pairs-English", "acc."),
    "short_average": ("Short Average", "acc."),
    "full_average": ("Full Average", "acc."),
}
SMALLER_IS_BETTER = defaultdict(lambda: False)
SMALLER_IS_BETTER.update(
    {
        "inference_time_average": True,
        "wikitext": True,
        # 'crows_pairs_english': True,
    }
)


# get original
original_row = df[df["lib"] == "original"].iloc[0]
df = df[df["lib"] != "original"]


for MODELS, name in [(MODELS_ALL, "all"), (MODELS_OUR, "our")]:
    print("MODELS", MODELS)

    df_subset = df[df["name"].isin(MODELS)]

    # plot
    fig, axs = plt.subplots(2, 4, figsize=(13, 6), dpi=300, sharex=False, sharey=False)
    fig.suptitle(f"{MODEL_NAME} pruning \\wo finetuning ({DEVICE_NAME})", fontsize=13, fontweight="bold")
    fig.tight_layout(h_pad=3.5, w_pad=2.5)
    fig.subplots_adjust(bottom=0.18, left=0.05, right=0.98, top=0.9)

    names_only_one_sample = df_subset.groupby("name").filter(lambda x: len(x) == 1)["name"].unique()
    for ax, metric in zip(axs.flat, METRICS):
        pretty_metric, pretty_value = PRETTY_NAMES_METRICS[metric]
        pretty_row, _ = PRETTY_NAMES_METRICS[ROW]

        # plot original line
        ax.axhline(original_row[metric], linestyle="--", label="Original", color="red", alpha=ORIGINAL_OPACITY)
        # ax.axvline(original_row[ROW], linestyle='--', color='red', alpha=ORIGINAL_OPACITY)

        # plot by rounding on x-axis and inference time on y-axis; hue by uniformity
        sns.lineplot(
            data=df_subset[~df_subset["name"].isin(names_only_one_sample)],
            x=ROW,
            y=metric,
            hue="name",
            style="name",
            dashes={
                k: (4, 2) if "our" not in k.lower() and "original" not in k.lower() else "" for k in df["name"].unique()
            },
            markers=True,
            ax=ax,
            palette="Set2",
        )
        if len(names_only_one_sample) > 0:
            sns.scatterplot(
                data=df_subset[df_subset["name"].isin(names_only_one_sample)],
                x=ROW,
                y=metric,
                hue="name",
                style="name",
                ax=ax,
                palette="viridis",
            )

        # set labels
        ax.set_title(f"{pretty_metric} ({pretty_value})")
        ax.set_xlabel(f"{pretty_row}")
        if pretty_value == "s":
            ax.set_ylabel("Seconds")
        elif pretty_value == "ppl.":
            ax.set_ylabel("Perplexity")
        elif pretty_value == "acc.":
            ax.set_ylabel("Accuracy")
        else:
            ax.set_ylabel(pretty_value)
        # ax.set_ylabel(f"{pretty_metric} ({pretty_value})")

        if SMALLER_IS_BETTER[metric]:
            ax.invert_yaxis()

        if metric in METRICS_LIMITS:
            ax.set_ylim(*METRICS_LIMITS[metric])

        ax.set_xlim(*ROW_LIMITS)

    # add single legend to the bottom
    try:
        handles, labels = axs[0, 0].get_legend_handles_labels()
    except IndexError:
        handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles=handles, labels=labels, loc="center right")
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.01))
    for ax in axs.flat:
        if legend := ax.get_legend():
            legend.remove()

    plt.savefig(Path(__file__).parent / f"compare-2024-jun-15-{name}.svg")
    plt.savefig(Path(__file__).parent / f"compare-2024-jun-15-{name}.png", dpi=600)
    plt.show()
