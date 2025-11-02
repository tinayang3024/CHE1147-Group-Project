import os, sys
# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from src.utils.io_utils import load_parquet  # uses rehydration logic

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# only predicting kcat_value and km_value for now as kcat_km is derived
TARGET_COLS = ["kcat_value", "km_value"]
REMOVING_COLS = [
    "kcat_km", "max_enzyme_similarity", "max_organism_similarity",
    "sequence_length", "pmid", "cid", "brenda_id", "uniprot_id",
    "sequence_source", "mol", "pH", "temperature"
]

def prepare_features_from_df(df: pd.DataFrame):
    """Extract numeric features and target arrays from the cleaned enzyme dataset."""
    X = df.drop(columns=REMOVING_COLS, errors="ignore") \
          .drop(columns=[c for c in TARGET_COLS if c in df.columns], errors="ignore") \
          .select_dtypes(include=[np.number])
    targets = {t: df[t].values for t in TARGET_COLS if t in df.columns}
    print(f"[Feature prep] X shape={X.shape}, targets={list(targets.keys())}")
    return X, targets


def train_and_evaluate(X, y, target_name, test_size=0.1, random_state=42):
    """Train and evaluate an XGBoost regressor for one target variable with log scaling."""
    # --- 1️⃣ filter and log-transform ---
    mask = y > 0
    X = X[mask]
    y = np.log10(y[mask].astype(float))
    if len(y) == 0:
        raise ValueError(f"No valid positive entries for {target_name}")

    # --- 2️⃣ train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[{target_name}] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- 3️⃣ train model ---
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=random_state,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    # --- 4️⃣ predict in log space ---
    y_pred = model.predict(X_test)

    # --- 5️⃣ metrics in log space ---
    mae_log = mean_absolute_error(y_test, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_log = r2_score(y_test, y_pred)

    print(f"\n[{target_name}] Evaluation (log space):")
    print(f"MAE: {mae_log:.4f} | RMSE: {rmse_log:.4f} | R²: {r2_log:.4f}")

    model_path = os.path.join(OUTPUT_DIR, f"xgb_{target_name}.joblib")
    joblib.dump(model, model_path)
    print(f"[Saved model] → {model_path}")

    return {"mae": mae_log, "rmse": rmse_log, "r2": r2_log, "model_path": model_path}


def main():
    print("=== Loading cleaned dataset ===")
    df = load_parquet("enzyme_clean.parquet")
    print(f"Loaded DataFrame shape: {df.shape}")

    print("=== Preparing features and targets ===")
    X, targets = prepare_features_from_df(df)

    results = {}
    for target_name, y in targets.items():
        print(f"\n=== Training model for: {target_name} ===")
        results[target_name] = train_and_evaluate(X, y, target_name)

    print("\n=== Training Summary (log space) ===")
    for k, v in results.items():
        print(f"{k}: MAE={v['mae']:.4f}, RMSE={v['rmse']:.4f}, R²={v['r2']:.4f}")


if __name__ == "__main__":
    main()
