"""
The source CSV already carries a `split` column (train/validation/test) built by
Prepare_dataset.ipynb using a fingerprint that included source/destination port and
protocol. The exported model-facing CSV drops ports (EXCLUDE_PORTS_FROM_MODEL=True),
so flows that only differed by port collapse into apparent duplicates once you
fingerprint on the reduced feature set alone -- this is what dataset_quality_audit.ipynb
measured as the 9.88% train/validation "overlap".

This script rebuilds the partition assignment by deduplicating on the *exact
model-facing feature vector* (the 77-column set the CNN/RNN/QuantumAI actually
consume): whichever partition a duplicate group originally had the plurality of votes in keeps the group; ties are
broken train > validation > test. This guarantees zero duplicate feature vectors
cross partitions in the space the models actually see, independent of the semantic
argument about the original (port-inclusive) grouping.

Output: results/split_manifest.csv (immutable, row_id-indexed) and
results/split_summary.json (class support, reassignment stats).
"""
import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Portable path resolution: override with the HYBRID_IDS_REPO environment
# variable if the reference repository lives somewhere other than the
# default relative location (a sibling of this manuscript/ directory).
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(os.environ.get(
    "HYBRID_IDS_REPO",
    PROJECT_ROOT.parent.parent / "Hybrid-Intrusion-Detection-System-main",
)).resolve()
SRC = REPO_ROOT / "attached_assets" / "cicids2017_leakage_safe_without_target_binary.csv"
OUT_DIR = PROJECT_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARTITION_PRIORITY = {"train": 0, "validation": 1, "test": 2}  # tie-break order


def label_to_binary(label: str) -> int:
    return 0 if str(label).strip().upper() == "BENIGN" else 1


def fingerprint_row(values: np.ndarray) -> str:
    # Round to 6 decimals to avoid float-repr noise, then hash the exact tuple.
    rounded = np.round(values.astype(np.float64), 6)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def main():
    df = pd.read_csv(SRC)
    assert {"split", "row_id", "label"}.issubset(df.columns)

    feature_cols = [c for c in df.columns if c not in ("split", "row_id", "label")]
    assert len(feature_cols) == 77, f"expected 77 feature columns, got {len(feature_cols)}"

    df["target_binary"] = df["label"].apply(label_to_binary)
    df["original_split"] = df["split"]

    feat_matrix = df[feature_cols].to_numpy()
    df["fingerprint"] = [fingerprint_row(row) for row in feat_matrix]

    print(f"Loaded {len(df)} rows, {len(feature_cols)} feature columns")
    print(f"Original partition sizes:\n{df['original_split'].value_counts()}")

    # ---- Verify no label conflicts within any fingerprint group -----------------
    label_conflicts = df.groupby("fingerprint")["target_binary"].nunique()
    n_conflicts = int((label_conflicts > 1).sum())
    print(f"Fingerprint groups with conflicting binary labels: {n_conflicts}")
    if n_conflicts > 0:
        conflict_fps = label_conflicts[label_conflicts > 1].index
        print("  WARNING: dropping rows in label-conflicting groups (ambiguous ground truth)")
        df = df[~df["fingerprint"].isin(conflict_fps)].copy()

    # ---- Group-aware reassignment: whole duplicate groups -> one partition -----
    groups = df.groupby("fingerprint")["original_split"]
    n_groups = groups.ngroups
    n_multi_partition_groups = 0
    n_rows_moved = 0

    new_split = df["original_split"].copy()

    for fp, sub in df.groupby("fingerprint"):
        splits_in_group = sub["original_split"]
        unique_splits = splits_in_group.unique()
        if len(unique_splits) <= 1:
            continue
        n_multi_partition_groups += 1
        counts = splits_in_group.value_counts()
        max_count = counts.max()
        candidates = [s for s in counts.index if counts[s] == max_count]
        # tie-break: train > validation > test
        winner = sorted(candidates, key=lambda s: PARTITION_PRIORITY[s])[0]
        idx = sub.index
        moved = (splits_in_group != winner).sum()
        n_rows_moved += int(moved)
        new_split.loc[idx] = winner

    df["partition"] = new_split

    print(f"\nDuplicate-feature-vector groups spanning >1 partition: {n_multi_partition_groups}")
    print(f"Rows reassigned to close leakage: {n_rows_moved}")
    print(f"\nFinal partition sizes:\n{df['partition'].value_counts()}")

    # ---- Assert zero cross-partition duplicates after reassignment -------------
    check = df.groupby("fingerprint")["partition"].nunique()
    assert (check <= 1).all(), "leakage remains: some fingerprint group still spans >1 partition"
    print("\nASSERTION PASSED: zero duplicate feature vectors cross partitions.")

    # ---- Class support table per partition --------------------------------------
    support = (
        df.groupby(["partition", "label"]).size().unstack(fill_value=0)
        .reindex(["train", "validation", "test"])
    )
    print(f"\nClass support per partition:\n{support}")

    binary_support = (
        df.groupby(["partition", "target_binary"]).size().unstack(fill_value=0)
        .reindex(["train", "validation", "test"])
    )
    print(f"\nBinary class support per partition:\n{binary_support}")

    # ---- Write manifest + summary ------------------------------------------------
    manifest_cols = ["row_id", "original_split", "partition", "label", "target_binary", "fingerprint"]
    df[manifest_cols].to_csv(OUT_DIR / "split_manifest.csv", index=False)

    summary = {
        "n_rows_total": int(len(df)),
        "n_rows_dropped_label_conflict": int(n_conflicts),
        "n_feature_columns": len(feature_cols),
        "n_fingerprint_groups": int(n_groups),
        "n_multi_partition_groups_before_fix": int(n_multi_partition_groups),
        "n_rows_reassigned": int(n_rows_moved),
        "partition_sizes": df["partition"].value_counts().to_dict(),
        "class_support": support.to_dict(orient="index"),
        "binary_class_support": {str(k): v for k, v in binary_support.to_dict(orient="index").items()},
        "tie_break_priority": "train > validation > test",
    }
    with open(OUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Also emit the full feature matrix keyed by row_id + final partition ---
    full = df[["row_id", "partition", "label", "target_binary"] + feature_cols]
    full.to_csv(OUT_DIR / "locked_dataset.csv", index=False)

    print(f"\nWrote:\n  {OUT_DIR/'split_manifest.csv'}\n  {OUT_DIR/'split_summary.json'}\n  {OUT_DIR/'locked_dataset.csv'}")


if __name__ == "__main__":
    main()
