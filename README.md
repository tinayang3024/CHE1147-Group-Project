# **CHE1147 Group Project — Enzyme Kinetic Parameter Prediction**

This repository contains the full workflow for predicting enzyme kinetic parameters ($k_{\mathrm{cat}}$ and $K_m$) using machine learning models built on the **EnzyExtract (2025)** dataset.
Our pipeline includes data preprocessing, feature engineering, model development (Random Forest, LightGBM, XGBoost), and exploratory analysis.

The project is implemented in **Python**, with a modular and reproducible structure designed for research, experimentation, and deployment.

---

## 🚀 **Project Overview**

Enzyme kinetic parameters such as **$k_{\mathrm{cat}}$** and **$K_m$** are essential in enzymology, metabolic modeling, and protein engineering.
However, existing prediction models often struggle due to limited datasets lacking **pH** and **temperature**, two crucial environmental factors.

This project builds a high-performing XGBoost-based regression framework using the enriched EnzyExtract dataset, leveraging extensive feature engineering (PubChem descriptors, sequence features, fingerprints, EC hierarchy, etc.) to deliver competitive model performance.

---

## 📁 **Repository Structure**

The repository follows a clean, logical structure commonly used in machine-learning and data-science projects:

```
CHE1147-Group-Project/
│
├── data/
│   ├── processed/              # Auto-generated processed datasets (.parquet, .npz, .joblib)
│   └── raw/                    # Raw input data (ignored in GitHub)
│
├── src/
│   ├── data_processing/        # Parsing, cleaning, PubChem features, fingerprints, EC extraction
│   ├── features/               # Feature preparation pipeline
│   ├── modeling/               # Model training scripts (v1/v2/v3)
│   ├── outliers/               # Outlier detection rules
│   ├── utils/                  # I/O utilities, caching, config loading
│   ├── viz/                    # Plot utilities
│   └── main.py                 # Main preprocessing pipeline → generates data/processed/*
│
├── notebooks/                  # Analysis, visualization, experimentation
│
├── models/                     # Trained models (ignored in GitHub due to file size)
│
├── logs/                       # Model training logs and performance summaries
│
├── figs/                       # Figures for reports/presentations
│
├── environment.yml             # Conda environment for full reproducibility
├── requirements.txt            # Pip packages (if needed)
└── README.md
```

---

## 🔧 **Environment Setup**

All dependencies are fully captured in **environment.yml**.

### **1. Create the conda environment**

```bash
conda env create -f environment.yml
conda activate chem1147
```

### **2. (Optional) Install pip-only packages**

```bash
pip install -r requirements.txt
```

### **Environment Includes**

* Python 3.10
* RDKit
* XGBoost / LightGBM
* Scikit-learn
* PyArrow
* BioPython
* PubChemPy
* NumPy / Pandas
* Matplotlib / Seaborn

This ensures the project is **fully reproducible** across machines.

---

## 🛠️ **Data Preprocessing Pipeline**

Raw data must be placed into:

```
data/raw/
```

To generate all cleaned feature-rich datasets required for model training, run:

```bash
python src/main.py
```

This script:

* Loads raw EnzyExtract data
* Extracts sequence descriptors
* Computes molecular fingerprints
* Fetches PubChem descriptors via CID
* Parses temperature & pH
* Generates NPZ + Parquet feature tables
* Outputs results into:

```
data/processed/
```

---

## 🤖 **Training the Models**

The best performing models (XGBoost-based) are trained using:

```bash
python src/modeling/train_model_v3.py
```

This script:

* Loads the processed dataset
* Applies log-scaling to $k_{\mathrm{cat}}$ and $K_m$
* Trains tuned XGBoost regressors
* Performs cross-validation
* Logs performance metrics (MAE, RMSE, R²)
* Saves models into:

```
models/
```

> ⚠️ Note: `models/` is **excluded from GitHub** because files are too large.

All training logs are automatically written to:

```
logs/
```

---

## 📊 **Analysis & Exploration**

Additional analysis can be found in:

```
notebooks/
```

These notebooks include:

* Exploratory Data Analysis (EDA)
* Feature importance analysis
* Parity plots
* Fingerprint PCA/UMAP visualizations
* Outlier inspection
* Model comparisons

This folder is ideal for experiment tracking and documentation.

---

## 📈 **Results**

Performance logs and tuning experiments are stored under:

```
logs/output_*.txt
```

These contain:

* Hyperparameter tuning history
* Performance comparisons across models
* Iterative improvements from feature engineering
* Final XGBoost performance summaries

---

## 🧩 **Reproducibility Notes**

* Data preprocessing is deterministic when using the same raw dataset
* A fixed random seed ensures consistent model training splits
* All dependencies are fully specified in `environment.yml`
* Processed datasets and model artifacts are reproducible on any machine

---

## 📮 **Contact & Contribution**

For questions, contributions, or suggestions, feel free to open an Issue or Pull Request in the repository.
