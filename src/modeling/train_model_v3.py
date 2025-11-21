import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import loguniform
import joblib
from src.utils.io_utils import load_parquet
from src.config import DROP_MISSING_THRESHOLD

OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COLS = ["kcat_value", "km_value"]
REMOVING_COLS = [
    "kcat_km", "max_enzyme_similarity", "max_organism_similarity",
    "sequence_length", "pmid","uniprot_id",
    "sequence_source", "mol", "pH", "temperature"
]

def prepare_features_from_df(df: pd.DataFrame):
    # try to skip fingerprints for speed
    X = df.drop(columns=REMOVING_COLS, errors="ignore") \
          .drop(columns=[c for c in TARGET_COLS if c in df.columns], errors="ignore") \
          .select_dtypes(include=[np.number])
    targets = {t: df[t].values for t in TARGET_COLS if t in df.columns}
    print(f"[Feature prep] X shape={X.shape}, targets={list(targets.keys())}")
    return X, targets


def tune_and_train(X, y, target_name, random_state=42):
    """Run quick randomized search to tune XGBoost hyperparameters."""
    mask = y > 0
    X, y = X[mask], np.log10(y[mask].astype(float))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    base_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=-1, enable_categorical=True)

    # param_grid = {
    #     "n_estimators": [300, 500, 700],
    #     "learning_rate": [0.01, 0.05, 0.1],
    #     "max_depth": [4, 6, 8],
    #     "subsample": [0.7, 0.8, 0.9],
    #     "colsample_bytree": [0.6, 0.7, 0.8],
    #     "reg_lambda": loguniform(0.001, 0.1, 10),
    #     "min_child_weight": [1],
    # }

    # log 8
    # param_grid = {
    #     "n_estimators": [700, 1000, 1500],
    #     "learning_rate": [0.04, 0.05, 0.06],
    #     "max_depth": [8, 10, 20],
    #     "subsample": [0.5, 0.6, 0.7],
    #     "colsample_bytree": [0.8, 0.85, 0.9, 0.95],
    #     "reg_lambda": loguniform(0.001, 0.1, 10),
    #     "min_child_weight": [1],
    # }
    
    # log 9
    # param_grid = {
    #     "n_estimators": [800, 1000, 1200],
    #     "learning_rate": [0.02, 0.03, 0.04],
    #     "max_depth": [18, 20, 22],
    #     "subsample": [0.55, 0.6, 0.65],
    #     "colsample_bytree": [0.6, 0.7, 0.8],
    #     "reg_lambda": loguniform(0.001, 0.1, 10),
    #     "min_child_weight": [1],
    # }

    # log 10
    param_grid = {
        "n_estimators": [1000],
        "learning_rate": [0.04],
        "max_depth": [20],
        "subsample": [0.6],
        "colsample_bytree": [0.8],
        "reg_lambda": loguniform(0.001, 0.1, 10),
        "min_child_weight": [1],
    }

    
    # param_grid = {
    #     "n_estimators": [700, 1000, 1500],
    #     "learning_rate": [0.04, 0.05, 0.06],
    #     "max_depth": [8, 10, 20],
    #     "subsample": [0.5, 0.6, 0.7],
    #     "colsample_bytree": [0.8, 0.85, 0.9, 0.95],
    #     "reg_lambda": loguniform(0.001, 0.0001, 0.1, 10),
    #     "min_child_weight": [1],
    # }

    # param_grid = {
    #     "n_estimators": [300, 500],
    #     "learning_rate": [0.01, 0.05],
    #     "max_depth": [4, 6],
    #     "subsample": [0.7, 0.8],
    #     "colsample_bytree": [0.7, 0.8, 0.9],
    #     "reg_lambda": loguniform(0.1, 10),
    #     "min_child_weight": [1, 3],
    # }
    # param_grid = {
    #     "n_estimators": [500],
    #     "learning_rate": [0.05],
    #     "max_depth": [6],
    #     "subsample": [0.7],
    #     "colsample_bytree": [0.9],
    #     "reg_lambda": loguniform(0.1, 10),
    #     "min_child_weight": [1],
    # }

    print(f"[{target_name}] Running parameter search...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=15,
        scoring="r2",
        cv=3,
        random_state=random_state,
        verbose=1,
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"[{target_name}] Best params: {search.best_params_}")

    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n[{target_name}] Tuned Model (log space):")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    model_path = os.path.join(OUTPUT_DIR, f"xgb_{target_name}_tuned.joblib")
    joblib.dump(best_model, model_path)
    print(f"[Saved model] → {model_path}")

    return {"mae": mae, "rmse": rmse, "r2": r2, "model_path": model_path, "best_params": search.best_params_}


def main():
    print("=== Loading cleaned dataset ===")
    df = load_parquet("enzyme_clean.parquet")

    # drop columns with too many missing values
    missing_ratio = df.isnull().mean()
    df = df.drop(columns=missing_ratio[missing_ratio > DROP_MISSING_THRESHOLD].index)

    X, targets = prepare_features_from_df(df)

    # Uncomment to peak at the dataset
    # print("X feature names:", X.columns.tolist())
    # print("X shape:", X.shape)
    # print("df feature names:", df.columns.tolist())


    results = {}
    for target_name, y in targets.items():
        print(f"\n=== Tuning & Training: {target_name} ===")
        results[target_name] = tune_and_train(X, y, target_name)

    print("\n=== Training Summary (log space) ===")
    for k, v in results.items():
        print(f"{k}: MAE={v['mae']:.4f}, RMSE={v['rmse']:.4f}, R²={v['r2']:.4f}")
        print(f"  → Best Params: {v['best_params']}")


if __name__ == "__main__":
    main()