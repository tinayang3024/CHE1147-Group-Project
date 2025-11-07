import os, sys
# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing.load_basic import load_and_basic_clean
from src.data_processing.temp_ph import add_temperature_ph
from src.data_processing.sequence_features import add_sequence_features
from src.data_processing.smiles_features import add_smiles_features
from src.data_processing.pubchem_features import add_pubchem_features
from src.outliers.rules import apply_outlier_strategy
from src.features.feature_prep import prepare_features
from src.utils.io_utils import (
    save_parquet,
    save_features_npz,
    save_meta,
)
from src.viz.plots import plot_corr_heatmap, plot_pca_explained_variance
from src.data_processing.brenda_features import expand_enzyme_ec

def main():
    # 1) load + basic cleaning
    print("[main] starting data processing pipeline...")
    df = load_and_basic_clean()

    # 2) temperature / pH
    print("[main] adding temperature and pH features...")
    df = add_temperature_ph(df)

    # 3) sequence features
    print("[main] adding sequence-based features...")
    df = add_sequence_features(df)

    # 4) smiles → descriptors + fingerprints
    print("[main] adding SMILES-based features...")
    df = add_smiles_features(df)

    # 4.5) pubchem and brenda features 
    print("[main] adding PubChem-based features...")
    df = add_pubchem_features(df)
    # print("[main] adding BRENDA-based features...")
    # df = add_brenda_features(df)
    print("[main] expanding enzyme EC numbers...")
    df = expand_enzyme_ec(df)

    # 5) outlier removal
    print("[main] applying outlier removal strategy...")
    df_clean = apply_outlier_strategy(df)

    # 6) feature prep
    print("[main] preparing features...")
    X, y, meta = prepare_features(df_clean)

    # 7) save cleaned data so we don’t run this again
    print("[main] saving processed data and features...")
    save_parquet(df_clean, "enzyme_clean.parquet")
    save_features_npz(X, y, meta, fname="enzyme_features.npz")
    save_meta(meta)

    # 8) (optional) quick plots
    print("[main] generating diagnostic plots...")
    plot_corr_heatmap(df_clean, meta["cont_cols_kept"], "Continuous descriptors (post)", "cont_corr_post.png")
    if meta["fp_pca"] is not None:
        plot_pca_explained_variance(meta["fp_pca"])

    print("[main] done.")


if __name__ == "__main__":
    main()
