import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, DataStructs, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FP_SIZE = 256
# FP_SIZE = 2048 # too large for the model
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)


def smiles_to_mol(s):
    try:
        return Chem.MolFromSmiles(s, sanitize=True)
    except Exception:
        return None


def mol_descriptors(mol):
    if mol is None:
        return pd.Series(
            {
                "desc_MW": np.nan,
                "desc_LogP": np.nan,
                "desc_NumHDonors": np.nan,
                "desc_NumHAcceptors": np.nan,
                "desc_TPSA": np.nan,
                "desc_RotBonds": np.nan,
                "desc_AromaticRingCount": np.nan,
                "desc_RingCount": np.nan,
                "desc_NumHeavyAtoms": np.nan,
                "desc_ExactMolWt": np.nan,
                "desc_FractionCSP3": np.nan,
                "desc_CalcNumAliphaticRings": np.nan,
                "desc_CalcNumSaturatedRings": np.nan,
                "desc_CalcNumHeteroatoms": np.nan,
                "desc_CalcNumAromaticHeterocycles": np.nan,
                "desc_FindMolChiralCenters": np.nan,
                "desc_polarizability": np.nan,
            }
        )
    return pd.Series(
        {
            "desc_MW": Descriptors.MolWt(mol),
            "desc_LogP": Descriptors.MolLogP(mol),
            "desc_NumHDonors": Descriptors.NumHDonors(mol),
            "desc_NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "desc_TPSA": Descriptors.TPSA(mol),
            "desc_RotBonds": Descriptors.NumRotatableBonds(mol),
            "desc_AromaticRingCount": Descriptors.NumAromaticRings(mol),
            "desc_RingCount": Descriptors.RingCount(mol),
            "desc_NumHeavyAtoms": Descriptors.HeavyAtomCount(mol),
            "desc_ExactMolWt": Descriptors.ExactMolWt(mol),
            "desc_FractionCSP3": Descriptors.FractionCSP3(mol),
            "desc_CalcNumAliphaticRings": Descriptors.NumAliphaticRings(mol),
            "desc_CalcNumSaturatedRings": Descriptors.NumSaturatedRings(mol),
            "desc_CalcNumHeteroatoms": Descriptors.NumHeteroatoms(mol),
            "desc_CalcNumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles(mol),
            "desc_FindMolChiralCenters": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
            "desc_polarizability": Descriptors.MolMR(mol),
        }
    )


def mol_fingerprint(mol, generator=morgan_gen, nbits=FP_SIZE):
    if mol is None:
        return np.zeros(nbits, dtype=int)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def add_smiles_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mol"] = df.get("smiles", pd.Series(index=df.index)).apply(smiles_to_mol)

    desc_df = df["mol"].apply(mol_descriptors)
    # Note: not applying log-scaling for now as models can handle raw values
    # # log-scaling only for heavy-tailed ones
    # desc_df["desc_log_MW"] = np.log10(desc_df["desc_MW"])
    # desc_df["desc_log_TPSA"] = np.log1p(desc_df["desc_TPSA"])
    # # drop original skewed cols, keep log ones
    # desc_df = desc_df.drop(columns=["desc_MW", "desc_TPSA"])

    fps = np.vstack(df["mol"].apply(mol_fingerprint))
    fp_df = pd.DataFrame(fps, columns=[f"fp_{i}" for i in range(fps.shape[1])])

    out = pd.concat(
        [df.reset_index(drop=True), desc_df.reset_index(drop=True), fp_df], axis=1
    )
    return out
