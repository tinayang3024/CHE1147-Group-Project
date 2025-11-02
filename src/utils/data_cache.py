# src/utils/data_cache.py
import os
import pandas as pd
import joblib
from typing import Any

class DataCache:
    """Handles saving and loading intermediate datasets or features."""

    def __init__(self, cache_dir="data/processed"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def save(self, obj: Any, name: str):
        path = os.path.join(self.cache_dir, f"{name}.joblib")
        joblib.dump(obj, path)
        print(f"[Cache] Saved → {path}")

    def load(self, name: str):
        path = os.path.join(self.cache_dir, f"{name}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Cache] No cache found: {path}")
        obj = joblib.load(path)
        print(f"[Cache] Loaded ← {path}")
        return obj

    def save_df(self, df: pd.DataFrame, name: str):
        path = os.path.join(self.cache_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"[Cache] Saved DataFrame → {path}")

    def load_df(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.cache_dir, f"{name}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[Cache] No cached DataFrame found: {path}")
        df = pd.read_parquet(path)
        print(f"[Cache] Loaded DataFrame ← {path}")
        return df
