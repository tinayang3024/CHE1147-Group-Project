"""
brenda_features.py
--------------------------------
Feature enrichment using BRENDA metadata joined via brenda_id.
This script fetches enzyme-related biochemical context (e.g., EC number, cofactors)
from a provided BRENDA metadata table and merges it with the main dataset.
"""

import os
import pandas as pd
from src.utils.io_utils import load_parquet
from src.config import PROCESSED_DIR, DATA_DIR
import numpy as np

def expand_enzyme_ec(df: pd.DataFrame, col: str = "enzyme_ecs") -> pd.DataFrame:
    """
    Expand EC numbers into 4 categorical columns.
    Safely handles missing/invalid EC entries.
    """
    # Split EC string into 4 levels
    ec_split = df[col].astype(str).str.split('.', expand=True)
    ec_split = ec_split.iloc[:, :4].fillna(np.nan)

    # Rename columns
    ec_split.columns = [f"{col}_level{i+1}" for i in range(4)]

    # Convert to numeric categories (Int32 to allow NaN)
    for c in ec_split.columns:
        ec_split[c] = pd.to_numeric(ec_split[c], errors="coerce").astype("Int32")

    # Concatenate with original DataFrame
    df_out = pd.concat([df, ec_split], axis=1)
    return df_out


def add_brenda_features(df: pd.DataFrame, brenda_meta_path: str = None) -> pd.DataFrame:
    """
    Enrich enzyme dataset with metadata from BRENDA.
    Parameters
    ----------
    df : pd.DataFrame
        Main enzyme dataset (must contain 'brenda_id').
    brenda_meta_path : str, optional
        Path to CSV or Parquet file containing BRENDA metadata.
        Expected columns: ['brenda_id', 'enzyme_ecs', 'cofactors', 'reaction_type', 'organism']
    Returns
    -------
    pd.DataFrame
        Enriched dataset with new BRENDA-based features.
    """
    if "brenda_id" not in df.columns:
        raise ValueError("brenda_id column is required in the input DataFrame.")

    # Default metadata file path
    if brenda_meta_path is None:
        brenda_meta_path = os.path.join(DATA_DIR, "external", "brenda_metadata.csv")

    if not os.path.exists(brenda_meta_path):
        raise FileNotFoundError(f"BRENDA metadata not found at {brenda_meta_path}")

    print(f"[BRENDA] Loading metadata from {brenda_meta_path}")
    brenda_meta = pd.read_csv(brenda_meta_path)

    # Clean and select relevant columns
    expected_cols = ["brenda_id", "enzyme_ecs", "cofactors", "reaction_type", "organism"]
    brenda_meta = brenda_meta[[c for c in expected_cols if c in brenda_meta.columns]]

    # Drop duplicates, keep most recent or complete entry
    brenda_meta = brenda_meta.drop_duplicates(subset=["brenda_id"])

    # Merge with main DataFrame
    df_merged = df.merge(brenda_meta, on="brenda_id", how="left")

    # Optional: One-hot encode EC number and cofactor presence
    if "enzyme_ecs" in df_merged.columns:
        df_merged["EC_prefix"] = df_merged["enzyme_ecs"].astype(str).str.split(".").str[0]
        df_merged = pd.get_dummies(df_merged, columns=["EC_prefix"], prefix="EC", dummy_na=True)

    if "cofactors" in df_merged.columns:
        df_merged["has_cofactor"] = df_merged["cofactors"].notna().astype(int)

    print(f"[BRENDA] Added {df_merged.shape[1] - df.shape[1]} new feature columns.")
    return df_merged


# if __name__ == "__main__":
#     # Example standalone usage
#     fname = "enzyme_clean.parquet"
#     df = load_parquet(fname)
#     df_enriched = add_brenda_features(df)
#     out_path = os.path.join(PROCESSED_DIR, "enzyme_enriched.parquet")
#     df_enriched.to_parquet(out_path, index=False)
#     print(f"[BRENDA] Enriched data saved → {out_path}")
