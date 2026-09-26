"""
Fig. 2: label distribution of the training dataset over publication years.
Usage: python visualization/Bargraphplot.py [--input data/real_dataset.csv]
"""

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", default=os.path.join(ROOT, "data", "real_dataset.csv"))
parser.add_argument("--output", default=os.path.join(ROOT, "visualization", "training_data_label_distribution.png"))
parser.add_argument("--show", action="store_true")
args = parser.parse_args()

# Load training dataset
df = pd.read_csv(args.input)

# Map labels
label_map = {'0': 'Survey', '1': 'Non-Survey'}
df['Label'] = df['Label'].astype(str).str.strip().map(label_map)
df = df[df['Label'].notna()]

# Clean and filter year
df = df[pd.to_numeric(df['Year'], errors='coerce').notnull()]
df['Year'] = df['Year'].astype(int)
df = df[df['Year'] >= 1950]

# Group by Year and Label
yearly_distribution = df.groupby(['Year', 'Label']).size().unstack(fill_value=0).sort_index()

# Plot
colors = {'Survey': '#2ca02c', 'Non-Survey': '#1f77b4'}
ax = yearly_distribution.plot(
    kind='bar',
    stacked=True,
    color=colors,
    figsize=(12, 6),
    width=0.8
)

# Formatting
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.title("Label Distribution in Model Training Dataset Over Time")
plt.legend(title="Label")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(args.output, dpi=300)
print(f"🖼️  Saved '{args.output}'")
if args.show:
    plt.show()
