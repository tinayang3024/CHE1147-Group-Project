import io
import requests
import pandas as pd
import pyarrow.parquet as pq

from src.config import DATA_URL, TARGET_COLS


def load_data_from_url(url: str) -> pd.DataFrame:
    resp = requests.get(url)
    resp.raise_for_status()
    return pq.read_table(io.BytesIO(resp.content)).to_pandas()


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # drop raw kcat/km text columns if present
    df = df.drop(columns=[c for c in ["kcat", "km"] if c in df.columns])

    # keep rows where parsed kinetic values exist
    df = df.dropna(subset=["kcat_value", "km_value"]).reset_index(drop=True)

    # drop columns with > 50% missing
    missing_ratio = df.isnull().mean()
    df = df.drop(columns=missing_ratio[missing_ratio > 0.5].index)

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
