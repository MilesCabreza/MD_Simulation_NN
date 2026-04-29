import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# Function to activate the esm3_env environment and run the embedding generation script
def generate_embeddings_from_sequence(sequence: str, heavy_chain: str, light_chain: str, name: str, output_dir: str):
    subprocess.run(
        f"source ~/.bashrc && conda activate esm3 && python /u/mcabreza/lai_project/esm/esm_embed_seq_finalmodel.py "
        f"{heavy_chain} {light_chain} {name} {output_dir}",
        shell=True,
        check=True
    )
    return


# Defining the Input Sequence for Embedding Script
sequence = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQRLEWIGYINPSNDYTKYNQKFKDRATLTADKSANTAYMELSSLRSEDTAVYYCARQGFPYWGQGTLVTVSSASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRVEIVLTQSPATLSLSPGERATLSCSASSSVSYMHWYQQKPGQAPRRWIYDTSKLASGVPARFSGSGSGTDYTLTISSLEPEDFAVYYCHQLSSDPFTFGGGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
heavy_chain = "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYWMHWVRQAPGQRLEWIGYINPSNDYTKYNQKFKDRATLTADKSANTAYMELSSLRSEDTAVYYCARQGFPYWGQGTLVTVSSASTKGPSVFPLAPCSRSTSESTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTKTYTCNVDHKPSNTKVDKRV"
light_chain = "EIVLTQSPATLSLSPGERATLSCSASSSVSYMHWYQQKPGQAPRRWIYDTSKLASGVPARFSGSGSGTDYTLTISSLEPEDFAVYYCHQLSSDPFTFGGGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
name = "TestAntibody"
output_dir = "/u/mcabreza/lai_project/test_embeddings"  # Ensure this path matches where you want to store the embeddings

# Define the function to load both Heavy and Light chain embeddings
def load_embeddings(heavy_chain_file: str, light_chain_file: str):
    # Load the Heavy chain embeddings
    heavy_chain_df = pd.read_csv(heavy_chain_file)
    
    # Load the Light chain embeddings
    light_chain_df = pd.read_csv(light_chain_file)
    
    # Combine the two DataFrames. Assuming they have the same structure.
    combined_df = pd.concat([heavy_chain_df, light_chain_df], ignore_index=True)
    
    return combined_df

# Set this flag to False if you don't want to regenerate embeddings
REGenerate_embeddings = True

if REGenerate_embeddings:
    generate_embeddings_from_sequence(sequence, heavy_chain, light_chain, name, output_dir)
    embedding_df = load_embeddings(
        "/u/mcabreza/lai_project/test_embeddings/TestAntibody_H_chain_embeddings.csv",
        "/u/mcabreza/lai_project/test_embeddings/TestAntibody_L_chain_embeddings.csv"
    )
else:
    # Load the embeddings for Heavy and Light chains separately and combine them
    embedding_df = load_embeddings(
        "/u/mcabreza/lai_project/test_embeddings/TestAntibody_H_chain_embeddings.csv",
        "/u/mcabreza/lai_project/test_embeddings/TestAntibody_L_chain_embeddings.csv"
    )

# Check the combined DataFrame
print(embedding_df.head())

# Define the model architecture (HurdleMLP or similar)
class HurdleMLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
        )
        self.head_cls = torch.nn.Linear(128, out_dim)  # logits for binding presence
        self.head_reg = torch.nn.Linear(128, out_dim)  # raw magnitude output

        self.softplus = torch.nn.Softplus()

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        logits = self.head_cls(h)                # (B, T)
        reg_raw = self.head_reg(h)               # (B, T)
        mag = -self.softplus(reg_raw)            # <= 0 always (negative magnitudes)
        return logits, mag

    @torch.no_grad()
    def predict(self, x: torch.Tensor, bind_thresh: float = 0.5):
        """
        Returns final physical predictions in the same space as y:
          - 0 if no binding
          - negative magnitude if binding
        """
        logits, mag = self.forward(x)
        prob = torch.sigmoid(logits)
        bind = (prob >= bind_thresh).float()
        yhat = bind * mag  # if bind=0 => 0; if bind=1 => negative mag
        return yhat, prob


# **Load the pretrained model**: Initialize the model architecture
in_dim = len(embedding_df.columns) - 2  # Assuming features are in columns excluding 'Residue_Index' and 'Chain'
out_dim = 14  # Number of functional groups (adjust as needed)
model = HurdleMLP(in_dim, out_dim)  # Create an instance of your model architecture

# **Load the model weights**:
model_weights = torch.load('final_model.pth')  # Load the model weights (state_dict)
model.load_state_dict(model_weights)  # Load the state_dict into the model
model.eval()  # Set the model to evaluation mode

# Prepare the input for prediction
X = torch.tensor(embedding_df.iloc[:, 2:].values, dtype=torch.float32)  # Exclude 'Chain' and 'Residue_Index'

with torch.no_grad():
    yhat, prob = model.predict(X, bind_thresh=0.5)

predictions = yhat.cpu().numpy()
probabilities = prob.cpu().numpy()

# **Save predictions to a separate folder**
predictions_folder = "/u/mcabreza/lai_project/predictions"  # Specify the folder to save predictions
os.makedirs(predictions_folder, exist_ok=True)

# Prepare predictions DataFrame
predictions_df = pd.DataFrame(predictions, columns=[f"Prediction_{i+1}" for i in range(predictions.shape[1])])
predictions_df["Residue_Index"] = embedding_df["Residue_Index"]
predictions_df["Chain"] = embedding_df["Chain"]

# Save the predictions to CSV
predictions_path = os.path.join(predictions_folder, f"{name}_predictions.csv")
predictions_df.to_csv(predictions_path, index=False)

print(f"Predictions saved to {predictions_path}")

# **Plot the results for each functional group**
functional_groups = [
    "acec", "apolar", "benc", "dmeo", "forn", "foro",
    "hbacc", "hbdon", "imin", "iminh", "mamn", "meoo",
    "prpc", "tipo"
]

# Assuming the predictions dataframe includes both Heavy and Light chains together
# First, split the predictions by chain in your `embedding_df`

# Separate the predictions for Heavy and Light chains
def split_predictions_by_chain(df, predictions):
    heavy_chain_predictions = predictions[df['Chain'] == 'H']
    light_chain_predictions = predictions[df['Chain'] == 'L']
    return heavy_chain_predictions, light_chain_predictions

# Call the split function
heavy_chain_predictions, light_chain_predictions = split_predictions_by_chain(embedding_df, predictions)

def plot_predictions_vs_actual(pred_df, actual_df, functional_groups, out_dir):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # rename prediction columns so they match actual column names
    pred_col_map = {f"Prediction_{i+1}": fg for i, fg in enumerate(functional_groups)}
    pred_df = pred_df.rename(columns=pred_col_map).copy()

    # actual file uses Residue_ID, but prediction file uses Residue_Index
    actual_df = actual_df.rename(columns={"Residue_ID": "Residue_Index"}).copy()

    for fg in functional_groups:
        for chain in ["H", "L"]:
            pred_chain = pred_df[pred_df["Chain"] == chain].sort_values("Residue_Index")
            actual_chain = actual_df[actual_df["Chain"] == chain].sort_values("Residue_Index")

            merged = pred_chain[["Residue_Index", "Chain", fg]].merge(
                actual_chain[["Residue_Index", "Chain", fg]],
                on=["Residue_Index", "Chain"],
                suffixes=("_pred", "_actual")
            )

            # Predicted binding stats
            pred_vals = merged[f"{fg}_pred"]
            pred_binding = (pred_vals < 0).sum()
            pred_nonbinding = (pred_vals == 0).sum()

            # Actual binding stats
            actual_vals = merged[f"{fg}_actual"]
            actual_binding = (actual_vals < 0).sum()
            actual_nonbinding = (actual_vals == 0).sum()

            if merged.empty:
                print(f"Skipping {fg} {chain}: no merged rows")
                continue

            plt.figure(figsize=(14, 5))

            plt.scatter(
                merged["Residue_Index"],
                merged[f"{fg}_pred"],
                s=40,
                alpha=0.9,
                label=f"Predicted (bind={pred_binding}, no-bind={pred_nonbinding})"
            )

            plt.scatter(
                merged["Residue_Index"],
                merged[f"{fg}_actual"],
                s=30,
                alpha=0.7,
                marker="x",
                label=f"Actual (bind={actual_binding}, no-bind={actual_nonbinding})"
            )

            plt.axhline(0, linewidth=1.5, alpha=0.6)
            plt.grid(True, alpha=0.2)

            plt.gca().invert_yaxis()

            plt.xlabel("Residue Index", fontsize=14)
            plt.ylabel("Binding Propensity", fontsize=14)
            plt.title(f"{fg} - {chain} chain", fontsize=16, pad=12)

            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            plt.legend(fontsize=12)

            save_path = os.path.join(out_dir, f"{fg}_{chain}_chain.png")
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close()

actual_df = pd.read_csv("/projects/bdtk/mcabreza/ResidueLevel_Outputs/Abiprubart_ResidueLevel.csv")
plot_predictions_vs_actual(
    pred_df=predictions_df,
    actual_df=actual_df,
    functional_groups=functional_groups,
    out_dir="/u/mcabreza/lai_project/predictions/plots/actualvspredicted"
)