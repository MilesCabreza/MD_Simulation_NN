"""
insert_alignment_gaps.py

Inserts gap rows into the per-residue ANN datasets (train, test, val)
so that the "Abb" column matches the gapped alignment sequences.

Edit the paths in the CONFIG section below, then run:
    python insert_alignment_gaps.py

Outputs (written to OUTPUT_DIR):
    train_merged_aligned.csv
    test_merged_aligned.csv
    val_merged_aligned.csv
"""

import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — edit these paths before running
# ---------------------------------------------------------------------------

ALIGNMENT_PATH = "alignment_dataset.csv"   # path to your alignment CSV
TRAIN_PATH     = "train_merged.csv"        # path to train split
TEST_PATH      = "test_merged.csv"         # path to test split
VAL_PATH       = "val_merged.csv"          # path to val split
OUTPUT_DIR     = "aligned_output"          # folder where results will be saved

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1.  Load & build the alignment map
# ---------------------------------------------------------------------------

def load_alignment(path: str) -> dict[str, str]:
    """
    Returns {antibody_name: gapped_sequence_string}.

    Assumes the CSV has a 'Name' column plus 4 sequence-fragment columns
    (whatever they are named).  The 4 fragments are concatenated in column
    order, spaces stripped.
    """
    df = pd.read_csv(path)

    # Identify fragment columns (everything except 'Name')
    frag_cols = [c for c in df.columns if c != "Name"]
    if len(frag_cols) != 4:
        raise ValueError(
            f"Expected exactly 4 sequence fragment columns besides 'Name', "
            f"found: {frag_cols}"
        )

    sequences = {}
    for _, row in df.iterrows():
        name = str(row["Name"]).strip()
        seq  = "".join(str(row[c]).strip() for c in frag_cols).replace(" ", "")
        sequences[name] = seq

    return sequences


# ---------------------------------------------------------------------------
# 2.  Insert gap rows into one dataset
# ---------------------------------------------------------------------------

def insert_gaps(df: pd.DataFrame, alignment: dict[str, str]) -> pd.DataFrame:
    """
    For every antibody present in both `df` and `alignment`:
      - Walk through the gapped alignment sequence character by character.
      - For residue characters  → emit the next real row from df.
      - For '-' characters       → emit a gap row (all NaN except 'Antibody').

    Antibodies not found in the alignment are passed through unchanged.
    """
    all_columns  = df.columns.tolist()
    result_parts = []

    for antibody, ab_df in df.groupby("Antibody", sort=False):
        if antibody not in alignment:
            # No alignment info — keep as-is
            print(f"  [WARN] '{antibody}' not found in alignment; kept unchanged.")
            result_parts.append(ab_df)
            continue

        gapped_seq  = alignment[antibody]
        real_rows   = ab_df.reset_index(drop=True)

        # Quick sanity check
        ungapped = gapped_seq.replace("-", "")
        abb_str  = "".join(real_rows["Abb"].astype(str).tolist())
        if ungapped != abb_str:
            print(
                f"  [WARN] '{antibody}': ungapped alignment ({ungapped[:30]}…) "
                f"≠ dataset sequence ({abb_str[:30]}…).  Proceeding anyway."
            )

        rows_out    = []
        real_idx    = 0

        for ch in gapped_seq:
            if ch == "-":
                gap_row = {col: np.nan for col in all_columns}
                gap_row["Antibody"] = antibody
                rows_out.append(gap_row)
            else:
                if real_idx >= len(real_rows):
                    print(
                        f"  [WARN] '{antibody}': ran out of real residues at "
                        f"alignment position {real_idx}. Extra gap chars will be NaN."
                    )
                    gap_row = {col: np.nan for col in all_columns}
                    gap_row["Antibody"] = antibody
                    rows_out.append(gap_row)
                else:
                    rows_out.append(real_rows.iloc[real_idx].to_dict())
                    real_idx += 1

        if real_idx < len(real_rows):
            print(
                f"  [WARN] '{antibody}': {len(real_rows) - real_idx} residue(s) "
                f"left over after alignment exhausted."
            )

        result_parts.append(pd.DataFrame(rows_out, columns=all_columns))

    return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3.  Main
# ---------------------------------------------------------------------------

def process_split(input_path: str, alignment: dict[str, str], out_path: str):
    print(f"\nProcessing: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns):,} columns.")

    aligned_df = insert_gaps(df, alignment)
    print(f"  After alignment insertion: {len(aligned_df):,} rows.")

    aligned_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading alignment …")
    alignment = load_alignment(ALIGNMENT_PATH)
    print(f"  Found {len(alignment)} antibodies in alignment file.")

    splits = {
        "train": TRAIN_PATH,
        "test":  TEST_PATH,
        "val":   VAL_PATH,
    }

    for split_name, in_path in splits.items():
        base     = os.path.basename(in_path).replace(".csv", "_aligned.csv")
        out_path = os.path.join(OUTPUT_DIR, base)
        process_split(in_path, alignment, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
