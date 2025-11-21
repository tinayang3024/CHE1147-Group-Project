import os
import pandas as pd
import numpy as np

from src.utils.io_utils import load_parquet
from src.config import PROCESSED_DIR, DATA_DIR, BRENDA_FILEPATH
from brendapyrser import BRENDA
import math

# ---------------------------------------------------------
# 1. (kept) EC expansion helper
#    NOTE: if your main pipeline already calls this, you
#    can skip calling it in add_brenda_features.
# ---------------------------------------------------------
def expand_enzyme_ec(df: pd.DataFrame, col: str = "enzyme_ecs") -> pd.DataFrame:
    """
    Original version assumes `col` is a string EC like "2.7.1.40".
    If you're now storing lists, call this BEFORE converting
    to lists, or adapt to pick the first EC.
    """
    ec_split = df[col].astype(str).str.split(".", expand=True)
    ec_split = ec_split.iloc[:, :4].fillna(np.nan)
    ec_split.columns = [f"{col}_level{i+1}" for i in range(4)]
    for c in ec_split.columns:
        ec_split[c] = pd.to_numeric(ec_split[c], errors="coerce").astype("Int32")
    return pd.concat([df, ec_split], axis=1)


# Note:
# The code segments below were not used during model training after consulting with the course TA
# as feeding BRENDA kcat and km (even at different environmental conditions) to the model is considered "cheating"
# ---------------------------------------------------------
# 2. extract features from a single BRENDA reaction
# ---------------------------------------------------------
def _agg_log(vals):
    if not vals:
        return (0.0, 0.0, 0.0)
    logs = [math.log10(v) for v in vals if v > 0]
    return (min(logs), max(logs), sum(logs) / len(logs))

def _extract_reaction_features(rxn) -> dict:
    """
    Extract numeric kinetic features from a BRENDApyrser reaction object.

    - Aggregates min / max / mean KM and Kcat.
    - Applies log10 transform to reduce skew (ignores nonpositive values).
    """
    if rxn is None:
        return {
            "brenda_has_reaction": 0,
            "brenda_km_min": 0.0,
            "brenda_km_max": 0.0,
            "brenda_km_mean": 0.0,
            "brenda_kcat_min": 0.0,
            "brenda_kcat_max": 0.0,
            "brenda_kcat_mean": 0.0,
        }

    km_vals, kcat_vals = [], []

    # --- collect KM ---
    for entries in (getattr(rxn, "KMvalues", {}) or {}).values():
        for e in entries or []:
            v = e.get("value")
            if v is not None:
                try:
                    v = float(v)
                    if v > 0:
                        km_vals.append(v)
                except (TypeError, ValueError):
                    continue

    # --- collect Kcat ---
    for entries in (getattr(rxn, "Kcatvalues", {}) or {}).values():
        for e in entries or []:
            v = e.get("value")
            if v is not None:
                try:
                    v = float(v)
                    if v > 0:
                        kcat_vals.append(v)
                except (TypeError, ValueError):
                    continue

    km_min, km_max, km_mean = _agg_log(km_vals)
    kcat_min, kcat_max, kcat_mean = _agg_log(kcat_vals)
    return {
        "brenda_has_reaction": 1,
        "brenda_km_min": km_min,
        "brenda_km_max": km_max,
        "brenda_km_mean": km_mean,
        "brenda_kcat_min": kcat_min,
        "brenda_kcat_max": kcat_max,
        "brenda_kcat_mean": kcat_mean,
    }


# ---------------------------------------------------------
# 3. Main enrichment function
# ---------------------------------------------------------
def add_brenda_features(
    df: pd.DataFrame,
    ec_col: str = "enzyme_ecs",
) -> pd.DataFrame:
    """
    Enrich the dataframe with BRENDA-derived features using BRENDApyrser.

    Parameters
    ----------
    df : pd.DataFrame
        Your main training dataframe, must contain `enzyme_ecs`
        where each cell is a LIST of EC strings.
    ec_col : str
        Name of the EC column in df.

    Returns
    -------
    pd.DataFrame
        df with new columns:
            - brenda_has_reaction
            - brenda_n_substrates
            - brenda_n_products
    """
    if ec_col not in df.columns:
        raise ValueError(f"{ec_col} column is required in the input DataFrame.")

    if not os.path.exists(BRENDA_FILEPATH):
        raise FileNotFoundError(
            f"BRENDA TXT not found at {BRENDA_FILEPATH}. Please download it from BRENDA website."
        )

    print(f"[BRENDA] Parsing BRENDA from {BRENDA_FILEPATH} ...")
    brenda = BRENDA(BRENDA_FILEPATH)

    # cache per EC to avoid re-querying
    ec_cache: dict[str, dict] = {}
    brenda_rows: list[dict] = []

    for ecs in df[ec_col]:
        # normalize to list
        if isinstance(ecs, list) or isinstance(ecs, object):
            ec_list = ecs
        elif isinstance(ecs, str) and ecs.strip() not in ("", "None", "nan", "NaN", "null"):
            if ";" in ecs:
                ec_list = [e.strip() for e in ecs.split(";") if e.strip()]
            elif "," in ecs:
                ec_list = [e.strip() for e in ecs.split(",") if e.strip()]
            else:
                ec_list = [ecs.strip()]
        else:
            ec_list = []

        # if no ECs for this row
        if ec_list is None or len(ec_list) == 0:
        # if not ec_list:
            brenda_rows.append(_extract_reaction_features(None))
            continue

        # collect features from ALL ECs in this row
        per_ec_feats = []
        for ec in ec_list:
            if ec in ec_cache:
                feats = ec_cache[ec]
            else:
                try:
                    print(f"[BRENDA] Fetching EC: {ec}")
                    rxn = brenda.reactions.get_by_id(ec)
                except Exception as e:
                    print(f"[BRENDA] Fetching failed for {ec}: {e}")
                    rxn = None

                feats = _extract_reaction_features(rxn)
                ec_cache[ec] = feats

            per_ec_feats.append(feats)

        # aggregate features across ECs for THIS ROW
        # strategy: take max – so if any EC has reaction/substrates, we keep it
        row_feats = {
            "brenda_has_reaction": max(f["brenda_has_reaction"] for f in per_ec_feats),
            "brenda_km_min": min(f["brenda_km_min"] for f in per_ec_feats),
            "brenda_km_max": max(f["brenda_km_max"] for f in per_ec_feats),
            "brenda_km_mean": sum(f["brenda_km_mean"] for f in per_ec_feats)/len(per_ec_feats),
            "brenda_kcat_min": min(f["brenda_kcat_min"] for f in per_ec_feats),
            "brenda_kcat_max": max(f["brenda_kcat_max"] for f in per_ec_feats),
            "brenda_kcat_mean": sum(f["brenda_kcat_mean"] for f in per_ec_feats)/len(per_ec_feats),
        }
        brenda_rows.append(row_feats)

    brenda_df = pd.DataFrame(brenda_rows, index=df.index)

    df_out = pd.concat([df, brenda_df], axis=1)

    print(f"[BRENDA] Added {df_out.shape[1] - df.shape[1]} new columns from BRENDA.")
    return df_out
