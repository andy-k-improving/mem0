import json
import argparse

import pandas as pd


parser = argparse.ArgumentParser(description="Generate test score")
parser.add_argument("--metrics_path", type=str, default="evaluation_metrics.json", help="Path of the metrics")
args = parser.parse_args()

# Load the evaluation metrics data
with open(args.metrics_path, "r") as f:
    data = json.load(f)

# Flatten the data into a list of question items
all_items = []
for key in data:
    all_items.extend(data[key])

# Convert to DataFrame
df = pd.DataFrame(all_items)

# Convert category to numeric type
df["category"] = pd.to_numeric(df["category"])

# Calculate mean scores by category
result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

# Add count of questions per category
result["count"] = df.groupby("category").size()

# Print the results
print("Mean Scores Per Category:")
print(result)

# Calculate overall means
overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

print("\nOverall Mean Scores:")
print(overall_means)
