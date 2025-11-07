import numpy as np
import pandas as pd
import ast
import re
import io
import requests
import pandas as pd
import pyarrow.parquet as pq

from src.config import DATA_URL, TARGET_COLS


def extract_first_int(val):
    """Extract the first integer from a list, NumPy array, or stringified list."""
    # Handle missing or list/array directly
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return pd.NA
        try:
            return int(val[0])
        except Exception:
            return pd.NA

    # Handle scalar missing values
    if val is None:
        return pd.NA
    try:
        if pd.isna(val):
            return pd.NA
    except Exception:
        # if val is array-like, skip
        return pd.NA

    # Handle string or numeric
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return int(parsed[0])
            elif isinstance(parsed, (int, float)):
                return int(parsed)
        except Exception:
            match = re.search(r"\d+", val)
            if match:
                return int(match.group())
    return pd.NA


def load_data_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url)
    resp.raise_for_status()
    return pq.read_table(io.BytesIO(resp.content)).to_pandas()


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # drop raw kcat/km text columns if present
    df = df.drop(columns=[c for c in ["kcat", "km"] if c in df.columns])

    # keep rows where parsed kinetic values exist
    df = df.dropna(subset=["kcat_value", "km_value"]).reset_index(drop=True)

    # drop columns with > 60% missing and print dropped cols
    # note: adjusted from 50% to 60% to keep enzyme_ecs which is important for brenda features
    missing_ratio = df.isnull().mean()
    dropped_cols = missing_ratio[missing_ratio > 0.6].index.tolist()
    if dropped_cols:
        print(f"[Load & basic clean] Dropping columns with >60% missing: {dropped_cols}")
        df = df.drop(columns=dropped_cols)

    # take first integer from cid-like columns
    if "cid" in df.columns:
        df["cid"] = df["cid"].apply(extract_first_int)
    if "brenda_id" in df.columns:
        df["brenda_id"] = df["brenda_id"].apply(extract_first_int)

    # similarity filtering (>=60 or NaN)
    if "max_enzyme_similarity" in df.columns:
        df = df[(df["max_enzyme_similarity"].isna()) | (df["max_enzyme_similarity"] >= 60)]
    if "max_organism_similarity" in df.columns:
        df = df[(df["max_organism_similarity"].isna()) | (df["max_organism_similarity"] >= 60)]

    print("[Load & basic clean] shape:", df.shape)
    return df


def load_and_basic_clean() -> pd.DataFrame:
    df = load_data_from_url(DATA_URL)
    df = basic_clean(df)
    return df

