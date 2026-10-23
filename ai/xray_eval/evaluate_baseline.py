
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xray_inference import predict_chest_xray

ROOT = Path("ai/xray_eval")
CSV_PATH = ROOT / "nih_sample.csv"
IMAGE_DIR = ROOT / "images"
OUTPUT_PATH = ROOT / "baseline_results.csv"

# NIH labels that correspond to model output labels.
LABEL_MAP = {
    "Atelectasis": "Atelectasis",
    "Consolidation": "Consolidation",
    "Infiltration": "Infiltration",
    "Pneumothorax": "Pneumothorax",
    "Edema": "Edema",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Effusion": "Effusion",
    "Pneumonia": "Pneumonia",
    "Pleural_Thickening": "Pleural_Thickening",
    "Cardiomegaly": "Cardiomegaly",
    "Nodule": "Nodule",
    "Mass": "Mass",
    "Hernia": "Hernia",
    "Lung Lesion": "Lung Lesion",
    "Fracture": "Fracture",
    "Lung Opacity": "Lung Opacity",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
}

with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    records = list(csv.DictReader(f))

model_rows = []
failures = []

print(f"Evaluating {len(records)} images on CPU...")

for index, row in enumerate(records, start=1):
    image_path = IMAGE_DIR / row["Image Index"]

    try:
        result = predict_chest_xray(str(image_path))
        findings = set(row["Finding Labels"].split("|"))

        record = {
            "image": row["Image Index"],
            "patient_id": row["Patient ID"],
            "nih_labels": row["Finding Labels"],
        }

        for label in LABEL_MAP:
            record[f"score_{label}"] = result["all_outputs"].get(
               label,
                np.nan
            )
            record[f"label_{label}"] = (
                1 if label in findings else 0
            )

        model_rows.append(record)

    except Exception as exc:
        failures.append((row["Image Index"], str(exc)))

    if index % 10 == 0 or index == len(records):
        print(f"Processed {index}/{len(records)}")

results = pd.DataFrame(model_rows)
results.to_csv(OUTPUT_PATH, index=False)

print("\nSaved raw outputs to:", OUTPUT_PATH)
print("Successfully evaluated:", len(results))
print("Failed:", len(failures))

if failures:
    print("First failures:", failures[:5])

# Preliminary metrics use only model scores for the matching label.
# These are NOT clinical validation metrics.
print("\nPreliminary per-label results")
for label in LABEL_MAP:
    y = results[f"label_{label}"].to_numpy(dtype=int)
    scores = results[f"score_{label}"].to_numpy(dtype=float)
    valid = np.isfinite(scores)

    y = y[valid]
    scores = scores[valid]

    if len(y) == 0 or len(np.unique(y)) < 2:
        print(f"{label}: insufficient score/label variation")
        continue

    auc = roc_auc_score(y, scores)

    print(
        f"{label}: n={len(y)}, "
        f"positive={int(y.sum())}, "
        f"negative={int(len(y) - y.sum())}, "
        f"AUC={auc:.3f}"
    )