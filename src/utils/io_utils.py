import os
import pandas as pd
import numpy as np
import joblib

from src.config import PROCESSED_DIR
from rdkit import Chem

def prepare_for_save(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert non-serializable columns (e.g. RDKit Mol objects) to safe formats
    before saving to Parquet or CSV.
    """
    df_copy = df.copy()
    if "mol" in df_copy.columns:
        # Convert RDKit Mol objects back to SMILES strings safely
        df_copy["smiles_serialized"] = df_copy["mol"].apply(
            lambda m: Chem.MolToSmiles(m) if m is not None else np.nan
        )
        df_copy.drop(columns=["mol"], inplace=True)
    return df_copy

def save_parquet(df: pd.DataFrame, fname: str) -> str:
    path = os.path.join(PROCESSED_DIR, fname)
    prepare_for_save(df).to_parquet(path, index=False)
    print(f"[IO] saved DataFrame → {path}")
    return path

def load_parquet(fname: str) -> pd.DataFrame:
    """Load a parquet file, convert SMILES strings back to Mol objects if present."""
    path = os.path.join(PROCESSED_DIR, fname)
    df = pd.read_parquet(path)
    print(f"[IO] loaded DataFrame ← {path} shape={df.shape}")

    # If serialized SMILES are present, reconstruct Mol objects
    if "smiles_serialized" in df.columns:
        df["mol"] = df["smiles_serialized"].apply(
            lambda s: Chem.MolFromSmiles(s) if pd.notna(s) else None
        )
        df.drop(columns=["smiles_serialized"], inplace=True)

    return df


def save_features_npz(X: np.ndarray, y: dict, meta: dict, fname: str = "features.npz"):
    path = os.path.join(PROCESSED_DIR, fname)
    # convert dict of arrays to something savable
    y_sanitized = {k: v for k, v in y.items() if v is not None}
    np.savez_compressed(path, X=X, **y_sanitized, meta=meta)
    print(f"[IO] saved features → {path}")
    return path


def save_meta(meta: dict, fname: str = "feature_meta.joblib"):
    path = os.path.join(PROCESSED_DIR, fname)
    joblib.dump(meta, path)
    print(f"[IO] saved meta → {path}")
    return path
