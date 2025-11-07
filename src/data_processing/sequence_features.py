import re
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import math

STD = set("ACDEFGHIKLMNPQRSTVWY")
REPLACEMENTS = {
    "U": "C",
    "O": "K",
    "B": "D",
    "Z": "E",
    "J": "L",
}
paren_pat = re.compile(r"\([^)]*\)")


HYDROPHOBIC = set("AVILMFWY")
POLAR = set("STNQ")
CHARGED_POS = set("KRH")
CHARGED_NEG = set("DE")

def shannon_entropy(freqs):
    # freqs: iterable of fractions that sum to 1
    return -sum(f * math.log2(f) for f in freqs if f > 0)

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


# def compute_protein_features(seq: str) -> Dict[str, Any]:
#     s_clean, unknown_frac, orig_len = normalize_sequence(seq)
#     if not s_clean or orig_len < 10:
#         return {}
#     pa = ProteinAnalysis(s_clean)
#     feats = {
#         "sequence_length": orig_len,
#         "clean_length": len(s_clean),
#         "unknown_frac": unknown_frac,
#         "aromaticity": pa.aromaticity(),
#         "instability_index": pa.instability_index(),
#         "isoelectric_point": pa.isoelectric_point(),
#         "gravy": pa.gravy(),
#     }
#     h, t, e = pa.secondary_structure_fraction()
#     feats.update({"frac_helix": h, "frac_turn": t, "frac_sheet": e})
#     for aa, frac in pa.amino_acids_percent.items():
#         feats[f"aa_{aa}"] = frac
#     return feats

def compute_protein_features(seq: str) -> Dict[str, Any]:
    # your normalize_sequence(...) assumed
    s_clean, unknown_frac, orig_len = normalize_sequence(seq)
    if not s_clean or orig_len < 10:
        return {}

    pa = ProteinAnalysis(s_clean)
    aa_percent = pa.amino_acids_percent  # dict AA -> fraction

    # basic existing features
    feats = {
        "sequence_length": orig_len,
        "clean_length": len(s_clean),
        "unknown_frac": unknown_frac,
        "aromaticity": pa.aromaticity(),
        "instability_index": pa.instability_index(),
        "isoelectric_point": pa.isoelectric_point(),
        "gravy": pa.gravy(),
        "mol_weight": pa.molecular_weight(),
    }

    # secondary structure fractions
    h, t, e = pa.secondary_structure_fraction()
    feats.update({"frac_helix": h, "frac_turn": t, "frac_sheet": e})

    # amino acid fractions (what you already do)
    for aa, frac in aa_percent.items():
        feats[f"aa_{aa}"] = frac

    # 1) grouped residues
    feats["frac_hydrophobic"] = sum(aa_percent.get(a, 0.0) for a in HYDROPHOBIC)
    feats["frac_polar"] = sum(aa_percent.get(a, 0.0) for a in POLAR)
    feats["frac_charged_pos"] = sum(aa_percent.get(a, 0.0) for a in CHARGED_POS)
    feats["frac_charged_neg"] = sum(aa_percent.get(a, 0.0) for a in CHARGED_NEG)

    # 2) net charge at pH 7 (very crude): (+)K,R,H minus (-)D,E counts
    pos_count = sum(s_clean.count(a) for a in CHARGED_POS)
    neg_count = sum(s_clean.count(a) for a in CHARGED_NEG)
    feats["net_charge_pH7_approx"] = pos_count - neg_count
    feats["charge_ratio_pos_neg"] = (pos_count + 1e-6) / (neg_count + 1e-6)

    # 3) aliphatic index
    # AI = %A + 2.9×%V + 3.9×(%I + %L)
    a = aa_percent.get("A", 0.0)
    v = aa_percent.get("V", 0.0)
    i_ = aa_percent.get("I", 0.0)
    l = aa_percent.get("L", 0.0)
    feats["aliphatic_index"] = a + 2.9 * v + 3.9 * (i_ + l)

    # 4) extinction coefficients
    ec_red, ec_ox = pa.molar_extinction_coefficient()
    feats["ext_coeff_reduced"] = ec_red
    feats["ext_coeff_oxidized"] = ec_ox

    # 5) sequence entropy (diversity of AA usage)
    feats["aa_entropy"] = shannon_entropy(aa_percent.values())

    # 6) optional: low-complexity-ish measure (very crude)
    runs = 0
    i = 0
    while i < len(s_clean):
        j = i + 1
        while j < len(s_clean) and s_clean[j] == s_clean[i]:
            j += 1
        run_len = j - i
        if run_len >= 3:
            runs += run_len
        i = j
    feats["low_complexity_frac"] = runs / len(s_clean)

    # OPTIONAL: dipeptides (commented to avoid 400 extra cols)
    # dipeptides = {f"di_{a}{b}": 0.0 for a in "ACDEFGHIKLMNPQRSTVWY" for b in "ACDEFGHIKLMNPQRSTVWY"}
    # for i in range(len(s_clean) - 1):
    #     di = s_clean[i:i+2]
    #     if di in dipeptides:
    #         dipeptides[f"di_{di}"] += 1
    # # normalize
    # total_di = max(len(s_clean) - 1, 1)
    # for k in dipeptides:
    #     dipeptides[k] /= total_di
    # feats.update(dipeptides)

    return feats


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    seq_series = df.get("sequence", pd.Series(index=df.index)).astype(str)
    seq_feats = [
        compute_protein_features(s) for s in tqdm(seq_series, desc="Sequence feats")
    ]
    seq_df = pd.DataFrame(seq_feats)
    return pd.concat([df.reset_index(drop=True), seq_df.reset_index(drop=True)], axis=1)
