import os

# --- global paths ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(ROOT_DIR, "figs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# make sure folders exist
for _d in [FIG_DIR, DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    os.makedirs(_d, exist_ok=True)

# --- data source ---
DATA_URL = (
    "https://github.com/ChemBioHTP/EnzyExtract/raw/main/data/export/TheData_kcat.parquet"
)

# --- feature / model configs ---
CORR_THRESHOLD = 0.90
FP_VAR_THRESHOLD = 0.01
USE_PCA_FOR_FP = True
FP_PCA_N_COMPONENTS = 200
RANDOM_STATE = 42

TARGET_COLS = ["kcat_km", "kcat_value", "km_value"]
