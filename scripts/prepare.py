import os
import glob
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

raw_dir = "data/raw"
output_dir = params["prepare"]["output_dir"]
os.makedirs(output_dir, exist_ok=True)

def load_files(pattern, label_name):
    files = glob.glob(os.path.join(raw_dir, pattern))
    dfs = []
    for file in files:
        # Load TSV (handling single text column vs text+label)
        df = pd.read_csv(file, sep="\t", header=None, names=["text"])
        df["label"] = label_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# Load positive and negative sets
train_pos = load_files("*train*pos*.tsv", "Positive")
train_neg = load_files("*train*neg*.tsv", "Negative")
test_pos = load_files("*test*pos*.tsv", "Positive")
test_neg = load_files("*test*neg*.tsv", "Negative")

train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=params["base"]["seed"])
test_df = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1, random_state=params["base"]["seed"])

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False, encoding="utf-8-sig")
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False, encoding="utf-8-sig")
print(f"Prepared dataset: {len(train_df)} train rows, {len(test_df)} test rows.")
