from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def domain_outlier_mask(df: pd.DataFrame) -> pd.Series:
    m = pd.Series(False, index=df.index)
    if "pH_value" in df:
        m |= (df["pH_value"] <= 0) | (df["pH_value"] > 14)
    if "temperature_C" in df:
        m |= (df["temperature_C"] < -10) | (df["temperature_C"] > 120)
    if "kcat_value" in df:
        m |= df["kcat_value"] <= 0
    if "km_value" in df:
        m |= df["km_value"] <= 0
    if "unknown_frac" in df:
        m |= df["unknown_frac"] > 0.2
    return m

def _log_values(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s[(s > 0) & np.isfinite(s)]
    return np.log10(s)


def kinetics_outlier_mask(
    df: pd.DataFrame,
    fence: float = 1.5,
    per_group: Optional[str] = "enzyme",
    target_rate: float = 0.05,
    fence_grid=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0),
    min_group_size: int = 50,
) -> pd.Series:
    have_kcat = "kcat_value" in df.columns
    have_km = "km_value" in df.columns
    if not (have_kcat and have_km):
        return pd.Series(False, index=df.index, dtype=bool)

    num = pd.to_numeric(df["kcat_value"], errors="coerce")
    den = pd.to_numeric(df["km_value"], errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
    ratio[~np.isfinite(ratio)] = np.nan

    groups = (
        [(None, df)]
        if not per_group or per_group not in df.columns
        else df.groupby(per_group, dropna=False)
    )
    final_mask = pd.Series(False, index=df.index, dtype=bool)

    for gname, sub in groups:
        idx = sub.index
        x_kcat = _log_values(sub["kcat_value"])
        x_km = _log_values(sub["km_value"])
        x_ratio = _log_values(ratio.loc[idx])

        def _iqr_mask(x_log: pd.Series, f: float):
            if x_log.empty:
                return pd.Series(False, index=x_log.index)
            q1, q3 = np.nanpercentile(x_log, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - f * iqr, q3 + f * iqr
            return (x_log < lo) | (x_log > hi)

        # adapt
        if len(idx) >= min_group_size:
            m_try = None
            for f in fence_grid:
                m = pd.Series(False, index=idx, dtype=bool)
                if not x_kcat.empty:
                    m |= _iqr_mask(x_kcat, f).reindex(idx, fill_value=False)
                if not x_km.empty:
                    m |= _iqr_mask(x_km, f).reindex(idx, fill_value=False)
                if not x_ratio.empty:
                    m |= _iqr_mask(x_ratio, f).reindex(idx, fill_value=False)
                if m.mean() <= target_rate:
                    m_try = m
                    break
            if m_try is None:
                m_try = pd.Series(False, index=idx, dtype=bool)
            final_mask.loc[idx] = m_try.values
        else:
            m = pd.Series(False, index=idx, dtype=bool)
            if not x_kcat.empty:
                m |= _iqr_mask(x_kcat, fence).reindex(idx, fill_value=False)
            if not x_km.empty:
                m |= _iqr_mask(x_km, fence).reindex(idx, fill_value=False)
            if not x_ratio.empty:
                m |= _iqr_mask(x_ratio, fence).reindex(idx, fill_value=False)
            final_mask.loc[idx] = m.values

    return final_mask.fillna(False).astype(bool)


def iso_forest_mask(
    df: pd.DataFrame, feature_cols: list, cont_rate=0.01, random_state=42
) -> pd.Series:
    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    if X.empty:
        return pd.Series(False, index=df.index)
    iso = IsolationForest(contamination=cont_rate, random_state=random_state)
    scores = iso.fit_predict(X)
    return pd.Series(scores == -1, index=df.index)


def apply_outlier_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m_dom = domain_outlier_mask(df)
    print(f"[Outliers] domain-rule drops: {m_dom.sum()}")
    df = df[~m_dom].copy()

    m_kin = kinetics_outlier_mask(df, fence=1.5, per_group=("enzyme" if "enzyme" in df.columns else None))
    print(f"[Outliers] kinetics log-IQR drops: {m_kin.sum()}")
    df = df[~m_kin].copy()

    cont_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cont_cols = [c for c in cont_cols if not c.startswith("fp_")]
    m_iso = iso_forest_mask(df, cont_cols, cont_rate=0.01)
    print(f"[Outliers] isolation-forest drops: {m_iso.sum()}")
    df = df[~m_iso].copy()

    return df
