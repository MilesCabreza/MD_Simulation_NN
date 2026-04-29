import torch
import pandas as pd
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
import os

# Load ESM3 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading ESM3 model on {device}...")
client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)
print("✅ Model loaded.\n")

def generate_embeddings(sequence: str, chain_label: str, name: str, output_dir: str) -> pd.DataFrame:
    """Generates embeddings for the given sequence and chain label and saves to CSV"""
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
    if emb.ndim == 3:
        emb = emb[0]

    emb = emb.detach().cpu().numpy()

    # Prepare the embedding DataFrame with residue indices
    df_chain = pd.DataFrame(emb)
    df_chain.insert(0, "Chain", chain_label)
    df_chain.insert(1, "Residue_Index", range(1, len(df_chain) + 1))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save embeddings to separate files based on the chain label
    output_path = os.path.join(output_dir, f"{name}_{chain_label}_chain_embeddings.csv")
    df_chain.to_csv(output_path, index=False)
    print(f"{chain_label.capitalize()} chain embeddings saved to {output_path}")
    return df_chain

def process_sequence(heavy_seq: str, light_seq: str, name: str, output_dir: str) -> pd.DataFrame:
    combined_df = pd.DataFrame()
    try:
        # Generate embeddings for heavy chain (H)
        df_H = generate_embeddings(heavy_seq, "H", name, output_dir)
        # Generate embeddings for light chain (L)
        df_L = generate_embeddings(light_seq, "L", name, output_dir)

        # Debugging: Check if both chains have data
        print(f"Heavy chain DataFrame shape: {df_H.shape}")
        print(f"Light chain DataFrame shape: {df_L.shape}")

        # Combine both heavy and light chain embeddings
        combined_df = pd.concat([df_H, df_L], ignore_index=True)
    except Exception as e:
        print(f"❌ Failed processing {name}: {e}")
    return combined_df

import sys

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python esm_embed_seq_finalmodel.py <heavy_seq> <light_seq> <name> <output_dir>")
        sys.exit(1)

    heavy_chain = sys.argv[1]
    light_chain = sys.argv[2]
    name = sys.argv[3]
    output_dir = sys.argv[4]

    df_embeddings = process_sequence(heavy_chain, light_chain, name, output_dir)
    print(f"Generated embeddings for {name}")

    