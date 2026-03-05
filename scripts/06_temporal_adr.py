"""
============================================================
 MedWatchPro - Temporal ADR Processing
 Purpose: Process FAERS data to compute time-to-onset for
          drug-side effect pairs, then generate temporal
          labels for our 100 side effects.
============================================================
"""

import pandas as pd
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAERS_DIR = os.path.join(BASE_DIR, "temp-FAERS")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

print("=" * 60)
print("  MedWatchPro - Temporal ADR Processing")
print("=" * 60)

# ── 1. Load our side effects ────────────────────────────
se_df = pd.read_csv(os.path.join(PROCESSED_DIR, "nodes_side_effects.csv"))
our_se = {s.lower(): i for i, s in enumerate(se_df["side_effect"])}
print(f"\nOur side effects: {len(our_se)}")

# ── 2. Load FAERS data ─────────────────────────────────
print("\nLoading FAERS summarized reports...")
summ = pd.read_csv(os.path.join(FAERS_DIR, "df_summarized.csv.gz"),
                    usecols=["primaryid", "start_dt", "event_dt", "mappedName"])
print(f"  Reports: {len(summ):,}")

print("Loading FAERS reactions...")
reac = pd.read_csv(os.path.join(FAERS_DIR, "df_REAC.csv.gz"))
print(f"  Reactions: {len(reac):,}")

# ── 3. Join: get drug + side effect + dates ─────────────
print("\nJoining reports with reactions...")
merged = summ.merge(reac, on="primaryid", how="inner")
print(f"  Merged rows: {len(merged):,}")

# ── 4. Filter to our 100 side effects ──────────────────
merged["pt_lower"] = merged["pt"].str.lower()
merged = merged[merged["pt_lower"].isin(our_se)]
print(f"  Rows matching our 100 SEs: {len(merged):,}")

# ── 5. Compute time-to-onset in days ───────────────────
print("\nComputing time-to-onset...")
# Convert dates
merged["start_dt"] = pd.to_numeric(merged["start_dt"], errors="coerce")
merged["event_dt"] = pd.to_numeric(merged["event_dt"], errors="coerce")

# Drop rows without both dates
valid = merged.dropna(subset=["start_dt", "event_dt"]).copy()
print(f"  Rows with both dates: {len(valid):,}")

# Parse YYYYMMDD format
valid["start_date"] = pd.to_datetime(valid["start_dt"].astype(int).astype(str),
                                      format="%Y%m%d", errors="coerce")
valid["event_date"] = pd.to_datetime(valid["event_dt"].astype(int).astype(str),
                                      format="%Y%m%d", errors="coerce")
valid = valid.dropna(subset=["start_date", "event_date"])

valid["onset_days"] = (valid["event_date"] - valid["start_date"]).dt.days
print(f"  Valid onset calculations: {len(valid):,}")

# Filter reasonable range (0 to 365 days)
valid = valid[(valid["onset_days"] >= 0) & (valid["onset_days"] <= 365)]
print(f"  Reasonable range (0-365d): {len(valid):,}")

# ── 6. Categorize into temporal bins ────────────────────
def onset_category(days):
    if days <= 1:
        return 0  # Acute: 0-24 hours
    elif days <= 7:
        return 1  # Early: 1-7 days
    elif days <= 28:
        return 2  # Delayed: 1-4 weeks
    elif days <= 180:
        return 3  # Late: 1-6 months
    else:
        return 4  # Chronic: 6+ months

CATEGORY_NAMES = ["acute", "early", "delayed", "late", "chronic"]

valid["onset_cat"] = valid["onset_days"].apply(onset_category)

print("\nOnset category distribution:")
for cat_idx, cat_name in enumerate(CATEGORY_NAMES):
    count = (valid["onset_cat"] == cat_idx).sum()
    print(f"  {cat_name:>10}: {count:>8,} ({100*count/len(valid):.1f}%)")

# ── 7. Compute median onset per side effect ─────────────
print("\nComputing median onset per side effect...")
se_onset = valid.groupby("pt_lower").agg(
    median_days=("onset_days", "median"),
    mean_days=("onset_days", "mean"),
    count=("onset_days", "count"),
    most_common_cat=("onset_cat", lambda x: x.mode().iloc[0] if len(x) > 0 else 2),
).reset_index()

print(f"  Side effects with temporal data: {len(se_onset)}")

# ── 8. Map to our 100 side effects ───────────────────────
print("\nMapping to our side effect indices...")
temporal_labels = {}
for _, row in se_onset.iterrows():
    se_name = row["pt_lower"]
    if se_name in our_se:
        se_idx = our_se[se_name]
        temporal_labels[se_idx] = {
            "name": se_name,
            "median_days": round(float(row["median_days"]), 1),
            "mean_days": round(float(row["mean_days"]), 1),
            "category": CATEGORY_NAMES[int(row["most_common_cat"])],
            "category_idx": int(row["most_common_cat"]),
            "report_count": int(row["count"]),
        }

# Fill missing with "delayed" as default
for se_name, se_idx in our_se.items():
    if se_idx not in temporal_labels:
        temporal_labels[se_idx] = {
            "name": se_name,
            "median_days": 14.0,
            "mean_days": 14.0,
            "category": "delayed",
            "category_idx": 2,
            "report_count": 0,
        }

print(f"  Total mapped: {len(temporal_labels)}/100")

# ── 9. Save results ─────────────────────────────────────
output_path = os.path.join(PROCESSED_DIR, "temporal_labels.json")
with open(output_path, "w") as f:
    json.dump(temporal_labels, f, indent=2)
print(f"\nSaved: {output_path}")

# Also save as numpy array for model training
temporal_array = np.zeros(100, dtype=np.int64)
for idx, info in temporal_labels.items():
    temporal_array[int(idx)] = info["category_idx"]
np.save(os.path.join(PROCESSED_DIR, "temporal_categories.npy"), temporal_array)
print(f"Saved: temporal_categories.npy (shape: {temporal_array.shape})")

# ── 10. Print sample results ────────────────────────────
print("\n" + "=" * 60)
print("Sample Temporal Predictions:")
print("=" * 60)
for idx in sorted(temporal_labels.keys(), key=lambda x: temporal_labels[x]["report_count"], reverse=True)[:15]:
    info = temporal_labels[idx]
    print(f"  {info['name']:>30s} | {info['category']:>8s} | median {info['median_days']:>6.1f}d | {info['report_count']:>6,} reports")

print("\nDone!")
