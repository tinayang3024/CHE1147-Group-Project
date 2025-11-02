import re
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis

STD = set("ACDEFGHIKLMNPQRSTVWY")
REPLACEMENTS = {
    "U": "C",
    "O": "K",
    "B": "D",
    "Z": "E",
    "J": "L",
}
paren_pat = re.compile(r"\([^)]*\)")


def normalize_sequence(seq: str) -> Tuple[str, float, int]:
    s = str(seq)
    s = paren_pat.sub("", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z]", "", s).upper()
    for k, v in REPLACEMENTS.items():
        s = s.replace(k, v)
    orig_len = len(s)
    s_clean = "".join(ch for ch in s if ch in STD)
    unknown_frac = (1 - len(s_clean) / orig_len) if orig_len else np.nan
    return s_clean, unknown_frac, orig_len


def compute_protein_features(seq: str) -> Dict[str, Any]:
    s_clean, unknown_frac, orig_len = normalize_sequence(seq)
    if not s_clean or orig_len < 10:
        return {}
    pa = ProteinAnalysis(s_clean)
    feats = {
        "sequence_length": orig_len,
        "clean_length": len(s_clean),
        "unknown_frac": unknown_frac,
        "aromaticity": pa.aromaticity(),
        "instability_index": pa.instability_index(),
        "isoelectric_point": pa.isoelectric_point(),
        "gravy": pa.gravy(),
    }
    h, t, e = pa.secondary_structure_fraction()
    feats.update({"frac_helix": h, "frac_turn": t, "frac_sheet": e})
    for aa, frac in pa.amino_acids_percent.items():
        feats[f"aa_{aa}"] = frac
    return feats


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    seq_series = df.get("sequence", pd.Series(index=df.index)).astype(str)
    seq_feats = [
        compute_protein_features(s) for s in tqdm(seq_series, desc="Sequence feats")
    ]
    seq_df = pd.DataFrame(seq_feats)
    return pd.concat([df.reset_index(drop=True), seq_df.reset_index(drop=True)], axis=1)
