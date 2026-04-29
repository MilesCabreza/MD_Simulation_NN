# sequence_csv = "/projects/bdtk/mcabreza/fab_sequences.xlsx - Sheet1 copy.csv"  # <- your CSV with Name, Heavy chain, Light chain

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
import torch
import os
import pandas as pd

# === CONFIGURATION ===
sequence_csv = "/projects/bdtk/mcabreza/fab_sequences.xlsx - Sheet1.csv"  # <- your CSV with Name, Heavy chain, Light chain
output_dir = "ESM3_EMBEDDINGS_PER_RESIDUE_FROM_SEQ"
os.makedirs(output_dir, exist_ok=True)

fragmap_folder = "/projects/bdtk/mcabreza/ResidueLevel_Outputs"  # optional validation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading ESM3 model on {device}...")
client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
print("✅ Model loaded.\n")

# === LOAD SEQUENCES ===
df_seq = pd.read_csv(sequence_csv)

required_cols = {"Name", "Heavy chain", "Light chain"}
missing = required_cols - set(df_seq.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# Helper to process one chain
def process_chain(sequence: str, chain_label: str, name: str) -> pd.DataFrame:
    sequence = sequence.replace(" ", "").strip()
    if not sequence:
        raise ValueError(f"Empty sequence for chain {chain_label} of {name}")

    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    output = client.forward_and_sample(
        protein_tensor,
        SamplingConfig(return_per_residue_embeddings=True)
    )

    emb = output.per_residue_embedding  # shape is [L, D] or [1, L, D] depending on version

    # If there's a batch dim, squeeze it
    if emb.ndim == 3:
        emb = emb[0]

    emb = emb.detach().cpu().numpy()

    seq_len = len(protein.sequence)
    emb_len = emb.shape[0]

    if emb_len != seq_len:
        extra = emb_len - seq_len
        print(f"   ⚠️ Chain {chain_label} of {name}: emb_len={emb_len}, seq_len={seq_len}, extra={extra}")
        # Common case: 2 extra tokens = BOS + EOS
        if extra == 2:
            emb = emb[1:-1, :]
        elif extra == 1:
            # e.g. only BOS at position 0
            emb = emb[1:, :]
        else:
            raise ValueError(
                f"Unexpected number of extra tokens for {name} chain {chain_label}: "
                f"emb_len={emb_len}, seq_len={seq_len}"
            )

    # Now emb.shape[0] should equal seq_len
    df_chain = pd.DataFrame(emb)
    df_chain.insert(0, "Chain", chain_label)
    df_chain.insert(1, "Residue_Index", range(1, len(df_chain) + 1))
    print(f"✅ Processed {name} chain {chain_label} ({len(df_chain)} residues, matches sequence)")
    return df_chain

# === MAIN LOOP OVER ANTIBODIES ===
for _, row in df_seq.iterrows():
    name = str(row["Name"]).strip()
    heavy_seq = str(row["Heavy chain"])
    light_seq = str(row["Light chain"])

    print(f"🔬 Processing {name}...")

    output_path = os.path.join(output_dir, f"{name}_combined_embeddings.csv")
    if os.path.exists(output_path):
        print(f"⏭️  Skipping {name} — embeddings already exist.")
        continue

    combined_df = pd.DataFrame()
    try:
        df_H = process_chain(heavy_seq, "H", name)
        df_L = process_chain(light_seq, "L", name)
        combined_df = pd.concat([df_H, df_L], ignore_index=True)
    except Exception as e:
        print(f"❌ Failed processing {name}: {e}")
        continue

    # === OPTIONAL: VALIDATE AGAINST RESIDUE-LEVEL FRAGMAP CSV ===
    residue_csv_path = os.path.join(fragmap_folder, f"{name}_ResidueLevel.csv")
    if os.path.exists(residue_csv_path):
        residue_df = pd.read_csv(residue_csv_path)
        expected_residues = len(residue_df)
        actual_residues = len(combined_df)

        if expected_residues != actual_residues:
            print("❌ RESIDUE COUNT MISMATCH!")
            print(f"   Antibody: {name}")
            print(f"   Fragmap residues: {expected_residues}")
            print(f"   ESM3 residues:    {actual_residues}")
            print("   ➤ Skipping save. Fix required.\n")
            continue
        else:
            print(f"✅ Residue count OK ({expected_residues} residues).")
    else:
        print(f"⚠️  No residue-level CSV found for {name}. Skipping validation.")

    combined_df.to_csv(output_path, index=False)
    print(f"💾 Saved combined CSV: {output_path}\n")