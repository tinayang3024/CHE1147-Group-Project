import ast
import pandas as pd
from tqdm import tqdm
import pubchempy as pcp
import time
import random

# Properties available on PubChem
PUBCHEM_PROPERTIES = [
    "MolecularFormula", "MolecularWeight", "ConnectivitySMILES", "SMILES", "InChI",
    "InChIKey", "IUPACName", "XLogP", "ExactMass", "MonoisotopicMass", "TPSA",
    "Complexity", "Charge", "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount",
    "HeavyAtomCount", "IsotopeAtomCount", "AtomStereoCount", "DefinedAtomStereoCount",
    "UndefinedAtomStereoCount", "BondStereoCount", "DefinedBondStereoCount",
    "UndefinedBondStereoCount", "CovalentUnitCount", "Volume3D", "XStericQuadrupole3D",
    "YStericQuadrupole3D", "ZStericQuadrupole3D", "FeatureCount3D",
    "FeatureAcceptorCount3D", "FeatureDonorCount3D", "FeatureAnionCount3D",
    "FeatureCationCount3D", "FeatureRingCount3D", "FeatureHydrophobeCount3D",
    "ConformerModelRMSD3D", "EffectiveRotorCount3D", "ConformerCount3D"
]

# Properties that are already in our dataset fetched through smiles feature
EXISTING_FEATURES = {
    "MolecularWeight", "TPSA", "HBondDonorCount", "HBondAcceptorCount", "RotatableBondCount", "SMILES"
}

# Filter to get only new properties
NEW_PROPERTIES = [p for p in PUBCHEM_PROPERTIES if p not in EXISTING_FEATURES]


def fetch_pubchem_properties(cid, max_retries=5, base_delay=2.0):
    """
    Fetch PubChem compound properties for a given CID using PubChemPy.
    Retries automatically on server busy (503) errors.
    """
    # Normalize CID
    if isinstance(cid, float) and cid.is_integer():
        cid = int(cid)
    elif isinstance(cid, str) and cid.endswith(".0"):
        cid = int(float(cid))
    else:
        cid = int(cid)

    properties = [
        "molecular_formula", "isomeric_smiles", "inchi", "inchikey", "iupac_name",
        "xlogp", "exact_mass", "monoisotopic_mass", "complexity", "charge",
        "heavy_atom_count", "isotope_atom_count", "atom_stereo_count",
        "defined_atom_stereo_count", "undefined_atom_stereo_count",
        "bond_stereo_count", "defined_bond_stereo_count", "undefined_bond_stereo_count",
        "covalent_unit_count", "volume_3d", "x_steric_quadrupole_3d",
        "y_steric_quadrupole_3d", "z_steric_quadrupole_3d", "feature_count_3d",
        "feature_acceptor_count_3d", "feature_donor_count_3d", "feature_anion_count_3d",
        "feature_cation_count_3d", "feature_ring_count_3d", "feature_hydrophobe_count_3d",
        "conformer_model_rmsd_3d", "effective_rotor_count_3d", "conformer_count_3d"
    ]

    for attempt in range(1, max_retries + 1):
        try:
            compound = pcp.Compound.from_cid(cid)
            return {prop: getattr(compound, prop, None) for prop in properties}

        except Exception as e:
            err_msg = str(e)

            # If it's a 503 "server busy" or similar, back off and retry
            if "ServerBusy" in err_msg or "Too many requests" in err_msg or "503" in err_msg:
                sleep_time = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                print(f"[WARN] CID {cid} → Server busy. Retry {attempt}/{max_retries} after {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue

            # For other errors, break early
            print(f"[WARN] CID {cid} → {e}")
            break

    # If all retries fail, return None-filled dict
    print(f"[ERROR] CID {cid} → failed after {max_retries} retries.")
    return {prop: None for prop in properties}


def _normalize_cid(val):
    """
    We now expect cid to be a single integer.
    But if some rows still have '[123, 456]' or '[789]', take the first.
    If it's NaN or unparseable, return None.
    """
    if pd.isna(val):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        # sometimes parquet reads ints as float if there was NaN
        return int(val)
    if isinstance(val, str):
        # try to parse stringified list
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and len(parsed) > 0:
                return int(parsed[0])
            # maybe it's just a string "123"
            return int(parsed)
        except Exception:
            # last resort: try casting directly
            try:
                return int(val)
            except Exception:
                return None
    # unknown type
    return None


def add_pubchem_features(df: pd.DataFrame, cid_col: str = "cid") -> pd.DataFrame:
    """Adds additional PubChem features to the DataFrame using CIDs."""
    if cid_col not in df.columns:
        raise ValueError(f"Column '{cid_col}' not found in dataframe.")

    # normalize all cids to single integers
    df = df.copy()
    df[cid_col] = df[cid_col].apply(_normalize_cid)

    unique_cids = [c for c in df[cid_col].dropna().unique().tolist() if c is not None]
    print(f"[INFO] Fetching PubChem features for {len(unique_cids)} unique CIDs...")

    records = []
    for cid in tqdm(unique_cids, desc="Querying PubChem"):
        features = fetch_pubchem_properties(cid)
        features[cid_col] = cid
        records.append(features)

    df_features = pd.DataFrame(records)
    print(f"[INFO] Retrieved PubChem data shape = {df_features.shape}")

    # Merge back into main dataframe
    df_merged = df.merge(df_features, on=cid_col, how="left")
    return df_merged
