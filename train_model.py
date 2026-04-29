#!/usr/bin/env python3
import os
import json
import random
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA


# ===== CONFIG =====
# Point this at the same OUT_DIR you used in build_merged_dataset.py
MERGED_DIR = "/u/mcabreza/lai_project/merged_dataset"

BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 20
RANDOM_SEED = 42  # should match meta.json (we read it, but keep as fallback)

# Where to write predictions (same content as your original script did)
PRED_OUT_PATH = "/projects/bdtk/mcabreza/ANS_debug/test_predictions.csv"


# ===== DATASET =====

class ResidueDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
        normalize_targets: bool = False,
        normalize_features: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        X = self.df[self.feature_cols].to_numpy(dtype=np.float32)
        y = self.df[self.target_cols].to_numpy(dtype=np.float32)

        # Optional feature normalization
        self.normalize_features = normalize_features
        if normalize_features:
            if feature_mean is None or feature_std is None:
                self.feature_mean = X.mean(axis=0)
                self.feature_std = X.std(axis=0) + 1e-8
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
            X = (X - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = None
            self.feature_std = None

        # Optional target normalization
        self.normalize_targets = normalize_targets
        if normalize_targets:
            if target_mean is None or target_std is None:
                self.target_mean = y.mean(axis=0)
                self.target_std = y.std(axis=0) + 1e-8
            else:
                self.target_mean = target_mean
                self.target_std = target_std
            y = (y - self.target_mean) / self.target_std
        else:
            self.target_mean = None
            self.target_std = None

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

        # For grouping later / debugging
        self.antibody = self.df["Antibody"].tolist()
        self.chain = self.df["Chain"].tolist()
        self.res_idx = self.df["Residue_Index"].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "Antibody": self.antibody[idx],
            "Chain": self.chain[idx],
            "Residue_Index": self.res_idx[idx],
        }


# ===== MODEL =====

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===== TRAIN / EVAL =====

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    loss_fn = nn.MSELoss()

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(n, 1)


def eval_residue_level(model, loader, device):
    model.eval()
    ys = []
    y_preds = []
    antibodies = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            yp = model(x)

            ys.append(y.cpu().numpy())
            y_preds.append(yp.cpu().numpy())
            antibodies.extend(batch["Antibody"])

    y_all = np.concatenate(ys, axis=0) if ys else np.empty((0, 0), dtype=np.float32)
    yp_all = np.concatenate(y_preds, axis=0) if y_preds else np.empty((0, 0), dtype=np.float32)

    mse = ((y_all - yp_all) ** 2).mean() if y_all.size else float("nan")
    mae = np.abs(y_all - yp_all).mean() if y_all.size else float("nan")

    # R^2 per target, then mean
    if y_all.size:
        ss_res = ((y_all - yp_all) ** 2).sum(axis=0)
        ss_tot = ((y_all - y_all.mean(axis=0)) ** 2).sum(axis=0)
        r2_per_target = 1 - ss_res / (ss_tot + 1e-8)
        r2_mean = float(np.nanmean(r2_per_target))
    else:
        r2_per_target = np.array([])
        r2_mean = float("nan")

    return {
        "mse": float(mse),
        "mae": float(mae),
        "r2_mean": float(r2_mean),
        "r2_per_target": r2_per_target,
        "antibodies": antibodies,
        "y_true": y_all,
        "y_pred": yp_all,
    }


def load_df(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def fit_pca_transform(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_feature_cols: list,
    out_dir: str,
    n_components: int = 128,
    use_incremental: bool = True,
    batch_size: int = 5000,
    whiten: bool = False,
):
    """
    Fits scaler + PCA on train only, transforms train/val/test.
    Returns: (train_df2, val_df2, test_df2, pca_feature_cols)
    Saves scaler+pca to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    scaler_path = os.path.join(out_dir, f"scaler_pca_{n_components}.joblib")
    pca_path = os.path.join(out_dir, f"pca_{n_components}.joblib")

    # ---- Fit scaler on train only ----
    X_train = train_df[base_feature_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_scaled = scaler.fit_transform(X_train)

    # ---- Fit PCA on train only ----
    if use_incremental:
        pca = IncrementalPCA(n_components=n_components, whiten=whiten, batch_size=batch_size)
        # incremental wants multiple partial_fit calls
        for start in range(0, X_train_scaled.shape[0], batch_size):
            pca.partial_fit(X_train_scaled[start:start + batch_size])
    else:
        pca = PCA(n_components=n_components, svd_solver="randomized", whiten=whiten, random_state=42)
        pca.fit(X_train_scaled)

    # Persist transformers
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        X = df[base_feature_cols].to_numpy(dtype=np.float32)
        Xs = scaler.transform(X)
        Z = pca.transform(Xs).astype(np.float32)  # (N, K)

        out = df.copy()
        for k in range(n_components):
            out[f"pca_{k:04d}"] = Z[:, k]
        return out

    train_out = transform(train_df)
    val_out = transform(val_df)
    test_out = transform(test_df)

    pca_cols = [f"pca_{k:04d}" for k in range(n_components)]
    return train_out, val_out, test_out, pca_cols


def add_small_structured_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Adds cheap features that help alongside PCA:
      - chain_len
      - pos_norm
      - sin/cos pos enc
      - chain one-hot
    """
    out = df.copy()

    chain_len = out.groupby(["Antibody", "Chain"])["Residue_Index"].transform("max")
    out["chain_len"] = chain_len.astype(np.float32)
    out["pos_norm"] = ((out["Residue_Index"] - 1) / (out["chain_len"] - 1).replace(0, 1)).astype(np.float32)

    for k in [1, 2, 4, 8]:
        out[f"pos_sin_{k}"] = np.sin(2 * np.pi * k * out["pos_norm"]).astype(np.float32)
        out[f"pos_cos_{k}"] = np.cos(2 * np.pi * k * out["pos_norm"]).astype(np.float32)

    chain_dummies = pd.get_dummies(out["Chain"], prefix="Chain", dtype=np.float32)
    out = pd.concat([out, chain_dummies], axis=1)

    engineered_cols = (
        ["chain_len", "pos_norm"]
        + [f"pos_sin_{k}" for k in [1, 2, 4, 8]]
        + [f"pos_cos_{k}" for k in [1, 2, 4, 8]]
        + list(chain_dummies.columns)
    )
    return out, engineered_cols


def main():
    # --- read metadata produced by build_merged_dataset.py ---
    meta_path = os.path.join(MERGED_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json at {meta_path}. Run build_merged_dataset.py first.")

    with open(meta_path, "r") as f:
        meta: Dict[str, Any] = json.load(f)

    seed = int(meta.get("random_seed", RANDOM_SEED))
    feature_cols: List[str] = list(meta["feature_cols"])
    target_cols: List[str] = list(meta["target_cols"])

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_path = meta["paths"]["train"]
    val_path = meta["paths"]["val"]
    test_path = meta["paths"]["test"]

    # --- load merged dataframes ---
    train_df = load_df(train_path)
    val_df = load_df(val_path)
    test_df = load_df(test_path)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # --- PCA compression on embeddings (train-only fit) ---
    # Choose K (start with 128; try 64/256 too)
    K = 128

    train_df, val_df, test_df, pca_cols = fit_pca_transform(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    base_feature_cols=feature_cols,   # your original 2500 dims
    out_dir=MERGED_DIR,
    n_components=K,
    use_incremental=True,             # safer on memory
    batch_size=5000,
    whiten=False,
    )

    # --- add small structured features (optional but usually helps) ---
    train_df, extra_cols = add_small_structured_features(train_df)
    val_df, _ = add_small_structured_features(val_df)
    test_df, _ = add_small_structured_features(test_df)

    # Final feature set is PCA comps + small extras
    feature_cols = pca_cols + extra_cols

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    # --- create datasets with normalization based on train ---
    train_dataset = ResidueDataset(train_df, feature_cols, target_cols)
    val_dataset = ResidueDataset(
        val_df,
        feature_cols,
        target_cols,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
    )
    test_dataset = ResidueDataset(
        test_df,
        feature_cols,
        target_cols,
        feature_mean=train_dataset.feature_mean,
        feature_std=train_dataset.feature_std,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- build model ---
    in_dim = len(feature_cols)
    out_dim = len(target_cols)
    model = MLP(in_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_mse = float("inf")
    best_state = None

    # --- training loop ---
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = eval_residue_level(model, val_loader, device)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_mse={val_metrics['mse']:.4f} | "
            f"val_r2={val_metrics['r2_mean']:.4f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_metrics": {k: v for k, v in val_metrics.items() if k not in ("y_true", "y_pred", "antibodies")},
            }

    # --- final test evaluation ---
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        print(f"\nLoaded best model from epoch {best_state['epoch']} (val_mse={best_val_mse:.4f})")

    test_metrics = eval_residue_level(model, test_loader, device)
    print("\n=== TEST RESULTS (residue-level) ===")
    print(f"MSE: {test_metrics['mse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"R2 mean across targets: {test_metrics['r2_mean']:.4f}")

    # --- write predictions aligned with test_df rows (shuffle=False preserves order) ---
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y_pred = model(x).cpu().numpy()
            all_preds.append(y_pred)

    y_pred_all = np.concatenate(all_preds, axis=0)  # (N_test_residues, n_targets)

    pred_df = test_df.copy().reset_index(drop=True)
    for j, col in enumerate(target_cols):
        pred_df[f"{col}_pred"] = y_pred_all[:, j]

    os.makedirs(os.path.dirname(PRED_OUT_PATH), exist_ok=True)
    pred_df.to_csv(PRED_OUT_PATH, index=False)
    print(f"\nSaved test predictions to {PRED_OUT_PATH}")


if __name__ == "__main__":
    main()
