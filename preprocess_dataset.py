import sys

import pandas as pd
import numpy as np
import shutil
import config
from config import Config
from task_input import encode_task
import json
import os

def remove_existing_preprocessed():
    if os.path.isdir("processed"):
        shutil.rmtree("processed")


remove_existing_preprocessed()

# === Load raw dataset ===
# current_folder=sys.argv[1]

# df = pd.read_csv(current_folder+"/"+"dataset.csv")
df = pd.read_csv("dataset.csv")
print("Starting preprocessing...")
print("dataset.csv")
Config.load()

# === Fill or drop missing values ===
df = df.dropna()  # or use df.fillna(value)

# === Fix boolean fields ===
df["remote_work"] = df["remote_work"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
df["blocker_flag"] = df["blocker_flag"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

# === Ensure numeric types ===
num_fields = [
    "story_points", "team_size", "num_dependencies", "estimated_hours",
    "sprint_day", "created_hour", "meetings_today",
    "avg_experience", "juniors", "mediors", "seniors", "tech_leads"
]

for col in num_fields:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# === Encode each row using your model-ready format ===
X = []
y = []

bad_rows = 0
for _, row in df.iterrows():
    try:
        task = row.drop(["Unnamed: 0", "predicted_duration_days"], errors="ignore").to_dict()
        features = encode_task(task)
        if features.shape[1] != 47:
            print(f"Feature vector length mismatch: {features.shape[1]}")
            bad_rows += 1
            continue
        X.append(features[0])
        y.append(row["predicted_duration_days"])
    except Exception as e:
        print(f"Skipped row due to error: {e}")
        bad_rows += 1

X = np.array(X)
y = np.array(y)


print(f"Encoded {len(X)} tasks. Skipped {bad_rows} bad rows.")

# === Optional: Apply Min-Max scaling to features ===
scaler = {
    "min": X.min(axis=0).tolist(),
    "max": X.max(axis=0).tolist()
}

# Avoid divide-by-zero
denom = np.array(scaler["max"]) - np.array(scaler["min"])
denom[denom == 0] = 1e-6

X_scaled = (X - scaler["min"]) / denom

os.mkdir("processed")

# === Save outputs ===
np.save("processed/X_scaled.npy", X_scaled)
np.save("processed/y.npy", y)
with open("processed/scaler.json", "w") as f:
    json.dump(scaler, f)

print("Preprocessing complete! Data saved to folder+/processed/")
