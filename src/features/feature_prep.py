import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from src.config import (
    TARGET_COLS,
    CORR_THRESHOLD,
    FP_VAR_THRESHOLD,
    USE_PCA_FOR_FP,
    FP_PCA_N_COMPONENTS,
    RANDOM_STATE,
)


def split_feature_blocks(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in TARGET_COLS]
    fp_cols = [c for c in num_cols if c.startswith("fp_")]
    cont_cols = [c for c in num_cols if c not in fp_cols]
    return cont_cols, fp_cols


def correlation_prune(df: pd.DataFrame, cols: list, thr: float):
    if not cols:
        return []
    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > thr)]
    kept = [c for c in cols if c not in to_drop]
    print(f"[Correlation prune] {len(cols)} → keep {len(kept)} (thr={thr})")
    return kept


def reduce_fps(
    df: pd.DataFrame,
    fp_cols: list,
    var_thr=0.01,
    use_pca=False,
    pca_n=256,
    rs=42,
):
    if not fp_cols:
        return np.empty((len(df), 0)), [], {"vt": None, "pca": None, "mask": None}
    X = df[fp_cols].fillna(0).astype(np.float32).values
    vt = VarianceThreshold(threshold=var_thr)
    Xr = vt.fit_transform(X)
    kept_mask = vt.get_support()
    kept_cols = [c for c, keep in zip(fp_cols, kept_mask) if keep]
    print(f"[FP variance] {len(fp_cols)} → keep {Xr.shape[1]} (thr={var_thr})")

    pca_model, names = None, kept_cols
    if use_pca and Xr.shape[1] > 0:
        pca_model = PCA(n_components=min(pca_n, Xr.shape[1]), random_state=rs)
        Xr = pca_model.fit_transform(Xr)
        names = [f"fp_pca_{i}" for i in range(Xr.shape[1])]
        print(f"[FP PCA] reduced to {Xr.shape[1]} comps")

    return Xr, names, {"vt": vt, "pca": pca_model, "mask": kept_mask}


def prepare_features(df_in: pd.DataFrame):
    df = df_in.copy()

    # targets
    y = {t: df[t].values if t in df.columns else None for t in TARGET_COLS}

    # split
    cont_cols, fp_cols = split_feature_blocks(df)

    # prune continuous
    cont_keep = correlation_prune(df, cont_cols, CORR_THRESHOLD)

    # impute + scale continuous
    cont_imp = SimpleImputer(strategy="median")
    cont_scaler = StandardScaler()
    X_cont = (
        cont_scaler.fit_transform(cont_imp.fit_transform(df[cont_keep]))
        if cont_keep
        else np.empty((len(df), 0))
    )

    # fingerprints
    X_fp, fp_names, fp_steps = reduce_fps(
        df,
        fp_cols,
        var_thr=FP_VAR_THRESHOLD,
        use_pca=USE_PCA_FOR_FP,
        pca_n=FP_PCA_N_COMPONENTS,
        rs=RANDOM_STATE,
    )

    # final X
    X = np.hstack([X_cont, X_fp]) if X_fp.size else X_cont
    meta = {
        "cont_cols_in": cont_cols,
        "cont_cols_kept": cont_keep,
        "fp_cols_in": fp_cols,
        "fp_cols_kept": fp_names,
        "imputer": cont_imp,
        "scaler": cont_scaler,
        "fp_var_selector": fp_steps["vt"],
        "fp_pca": fp_steps["pca"],
        "fp_mask": fp_steps.get("mask"),
    }
    print(f"[Final features] X shape: {X.shape}")
    return X, y, meta
