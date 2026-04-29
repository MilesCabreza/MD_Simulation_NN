#!/usr/bin/env python3
import os
import json
import random
from typing import List, Tuple, Dict



import pandas as pd


# ===== CONFIG =====
EMB_DIR = "/u/mcabreza/lai_project/esm/ESM3_EMBEDDINGS_PER_RESIDUE_FROM_SEQ"    # *_combined_embeddings.csv
RES_DIR = "/projects/bdtk/mcabreza/ResidueLevel_Outputs"                       # *_ResidueLevel.csv

RANDOM_SEED = 42

TARGET_COLS = [
    "acec", "apolar", "benc", "dmeo", "forn", "foro",
    "hbacc", "hbdon", "imin", "iminh", "mamn", "meoo",
    "prpc", "tipo",
]

# Where to save merged outputs + metadata
OUT_DIR = "/u/mcabreza/lai_project/merged_dataset"


# ===== UTILITIES =====

def get_antibody_names(emb_dir: str, res_dir: str) -> List[str]:
    """Return sorted list of antibodies that have BOTH embeddings and residue-level data."""
    emb_names = set()
    res_names = set()

    for f in os.listdir(emb_dir):
        if f.endswith("_combined_embeddings.csv"):
            emb_names.add(f.replace("_combined_embeddings.csv", ""))

    for f in os.listdir(res_dir):
        if f.endswith("_ResidueLevel.csv"):
            res_names.add(f.replace("_ResidueLevel.csv", ""))

    return sorted(emb_names & res_names)


def split_antibodies(
    names: List[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """Randomly split antibody names into train/val/test."""
    rng = random.Random(seed)
    names = names[:]  # copy
    rng.shuffle(names)

    n = len(names)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = names[:n_train]
    val = names[n_train:n_train + n_val]
    test = names[n_train + n_val:]
    return train, val, test


def load_and_merge_antibody(
    name: str,
    emb_dir: str,
    res_dir: str,
) -> pd.DataFrame:
    """Load one antibody's embedding and residue-level CSV and merge per residue."""
    emb_path = os.path.join(emb_dir, f"{name}_combined_embeddings.csv")
    res_path = os.path.join(res_dir, f"{name}_ResidueLevel.csv")

    emb_df = pd.read_csv(emb_path)
    res_df = pd.read_csv(res_path)

    # Create Residue_Index per chain (1..L)
    res_df = res_df.copy()
    res_df["Residue_Index"] = res_df.groupby("Chain").cumcount() + 1

    # Inner join on Chain + Residue_Index
    merged = pd.merge(
        emb_df,
        res_df,
        on=["Chain", "Residue_Index"],
        how="inner",
        suffixes=("_emb", "_res"),
    )

    merged["Antibody"] = name

    # Sanity check: ensure we didn't lose residues (same check as original)
    if len(merged) != len(res_df):
        print(
            f"⚠️  Warning: merged rows != residue rows for {name}: "
            f"{len(merged)} vs {len(res_df)}"
        )

    return merged


def build_merged_dataset(
    antibody_names: List[str],
    emb_dir: str,
    res_dir: str,
) -> pd.DataFrame:
    """Concatenate merged per-residue data for a list of antibodies."""
    dfs = []
    for name in antibody_names:
        dfs.append(load_and_merge_antibody(name, emb_dir, res_dir))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def infer_feature_cols(emb_dir: str, antibody_name: str) -> List[str]:
    """Feature columns = all embedding dims (exclude join keys)."""
    emb_path = os.path.join(emb_dir, f"{antibody_name}_combined_embeddings.csv")
    example_emb = pd.read_csv(emb_path)
    return [c for c in example_emb.columns if c not in ["Chain", "Residue_Index"]]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Save as Parquet if possible (smaller/faster), else CSV.
    Use .parquet extension to trigger parquet.
    """
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    random.seed(RANDOM_SEED)

    ensure_dir(OUT_DIR)

    # --- 1) discover antibodies ---
    names = get_antibody_names(EMB_DIR, RES_DIR)
    print(f"Found {len(names)} antibodies with both embeddings and ResidueLevel.")
    if not names:
        raise RuntimeError("No antibodies found with both embedding + ResidueLevel files.")

    train_names, val_names, test_names = split_antibodies(names, seed=RANDOM_SEED)
    print("Train:", train_names)
    print("Val:  ", val_names)
    print("Test: ", test_names)

    # --- 2) infer feature columns from one train antibody ---
    feature_cols = infer_feature_cols(EMB_DIR, train_names[0])

    # --- 3) build merged dataframes ---
    train_df = build_merged_dataset(train_names, EMB_DIR, RES_DIR)
    val_df = build_merged_dataset(val_names, EMB_DIR, RES_DIR)
    test_df = build_merged_dataset(test_names, EMB_DIR, RES_DIR)

    # --- 4) write outputs ---
    # Prefer parquet (fast + compact). If your environment lacks parquet deps,
    # change extension to .csv.
    train_path = os.path.join(OUT_DIR, "train_merged.csv")
    val_path = os.path.join(OUT_DIR, "val_merged.csv")
    test_path = os.path.join(OUT_DIR, "test_merged.csv")

    save_dataframe(train_df, train_path)
    save_dataframe(val_df, val_path)
    save_dataframe(test_df, test_path)

    meta: Dict[str, object] = {
        "random_seed": RANDOM_SEED,
        "emb_dir": EMB_DIR,
        "res_dir": RES_DIR,
        "train_names": train_names,
        "val_names": val_names,
        "test_names": test_names,
        "feature_cols": feature_cols,
        "target_cols": TARGET_COLS,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "format": "parquet",
        "paths": {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        },
    }

    meta_path = os.path.join(OUT_DIR, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved merged datasets:")
    print(" -", train_path)
    print(" -", val_path)
    print(" -", test_path)
    print("Saved metadata:")
    print(" -", meta_path)


if __name__ == "__main__":
    main()
