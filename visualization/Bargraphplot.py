"""
Fig. 2: label distribution of the training dataset over publication years.
Usage: python visualization/Bargraphplot.py [--input data/real_dataset.csv]

Sized for one IEEE column; the research bars are hatched so the figure also reads in grayscale.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", default=os.path.join(ROOT, "data", "real_dataset.csv"))
parser.add_argument("--output", default=os.path.join(ROOT, "visualization", "training_data_label_distribution.png"))
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

df = pd.read_csv(args.input)
df = df[df["Label"].isin([0, 1])]
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"]).astype({"Year": int})
counts = df.groupby(["Year", "Label"]).size().unstack(fill_value=0).sort_index()
years = counts.index.values

SURVEY, RESEARCH = "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "hatch.linewidth": 0.6})

fig, ax = plt.subplots(figsize=(3.5, 2.1))
x = np.arange(len(years))
w = 0.4
gap = 0.02  # small surface gap between the paired bars
ax.bar(x - w / 2, counts[0].values, width=w - gap, color=SURVEY, label="Survey", zorder=3)
ax.bar(x + w / 2, counts[1].values, width=w - gap, color=RESEARCH, hatch="////", edgecolor="white",
       linewidth=0, label="Non-survey (research)", zorder=3)

ax.set_xticks(x)
ax.set_xticklabels([str(y) if i % 2 == 0 else "" for i, y in enumerate(years)], color=TEXT)
ax.set_xlabel("Publication year", color=TEXT)
ax.set_ylabel("Papers", color=TEXT)
ax.tick_params(axis="both", colors=MUTED, length=2)
ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
    ax.spines[side].set_linewidth(0.6)
ax.legend(frameon=False, loc="upper left", handlelength=1.2, labelcolor=TEXT)
ax.set_xlim(-0.6, len(years) - 0.4)

fig.tight_layout(pad=0.3)
fig.savefig(args.output, dpi=300)
print(f"🖼️  Saved '{args.output}' ({len(df):,} papers, {years.min()}–{years.max()})")
if args.show:
    plt.show()
