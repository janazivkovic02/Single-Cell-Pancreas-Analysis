# Single-Cell Pancreas Analysis
**Classification, Clustering and Association Rule Mining on Single-Cell RNA-seq Data**

## Overview

This project analyzes single-cell RNA sequencing (scRNA-seq) data from the **Human Pancreas
Single-Cell Transcriptomic Atlas** using several data mining and machine learning techniques.

The goal is to explore the structure of the data and evaluate different approaches for:

- **Association rule mining** on gene expression
- **Clustering of cells**
- **Classification of cell types**

The analysis follows a standard scRNA-seq preprocessing pipeline and then applies several
machine learning methods to understand both the **biological structure of the dataset** and
the **behavior of different algorithms**.

---

## Dataset

The analysis uses the dataset **A Single-Cell Transcriptomic Atlas of the Human and Mouse
Pancreas**, which contains UMI counts for thousands of genes measured in individual
pancreatic cells.

Files used in this project:

```
GSM2230757_human1_umifm_counts.csv.gz
GSM2230758_human2_umifm_counts.csv.gz
GSM2230759_human3_umifm_counts.csv.gz
GSM2230760_human4_umifm_counts.csv.gz
```

Place these files under `data/raw/`.

---

## Project Structure

```
Single-Cell-Pancreas-Analysis/
├── data/
│   └── raw/                  # input *.csv.gz count matrices (git-ignored)
├── main.ipynb                # full analysis pipeline
├── outputs/                  # generated figures / AnnData (git-ignored)
│   ├── figures/
│   └── adata/
├── src/
│   ├── __init__.py
│   ├── config.py             # paths, random seed, shared column names
│   ├── io_qc.py              # loading, quality control, normalization, HVG selection
│   ├── rules.py              # FP-Growth association rule mining + filtering
│   ├── genes_clustering.py   # gene co-expression network + module detection
│   ├── cells_clustering.py   # KMeans / GMM / Spectral / HDBSCAN / Leiden + silhouette
│   ├── classification.py     # cross-validated cell-type classification
│   └── plotting.py           # gene network visualization
├── pyproject.toml
└── README.md
```

### Module overview

| Module | Responsibility |
|---|---|
| `config.py` | Project paths, `RANDOM_STATE`, and shared AnnData key names. |
| `io_qc.py` | CSV → AnnData loading, QC metrics, MAD-based outlier flagging, normalization/log, highly variable gene selection. |
| `rules.py` | Binarize expression per cluster, mine frequent itemsets (FP-Growth), build and filter association rules. |
| `genes_clustering.py` | Turn rules into a gene network and detect gene modules via greedy modularity. |
| `cells_clustering.py` | Cluster cells with five algorithms and compare them by silhouette score. |
| `classification.py` | Stratified k-fold cross-validation of several classifiers on PCA/SVD features. |
| `plotting.py` | Draw the gene-module network. |

---

## Data Preprocessing

The preprocessing pipeline follows standard scRNA-seq practices:

1. Load raw UMI count matrices
2. Quality control and outlier detection (median-absolute-deviation rule)
3. Normalization and log transformation
4. Selection of highly variable genes (HVG)
5. Dimensionality reduction using PCA

The processed dataset is stored as an **AnnData object** (`outputs/adata/pancreas_processed.h5ad`).

---

## Association Rule Mining

Association rules are applied to gene expression patterns to identify **co-expressed genes**:

1. Selection of highly variable genes
2. Binarization of gene expression per cell cluster
3. Frequent itemset mining (FP-Growth)
4. Rule filtering using support, confidence and lift
5. Construction of a **gene association network**

Gene modules are detected by clustering the resulting gene network, revealing groups of
genes with similar expression patterns across cell populations.

---

## Cell Clustering

Several clustering algorithms are evaluated on the PCA representation of the dataset:

- **KMeans**
- **Gaussian Mixture Models (GMM)**
- **Spectral Clustering**
- **HDBSCAN**
- **Leiden community detection**

Results are compared using the **silhouette score**. KMeans, GMM and Spectral require a
preset number of clusters; Leiden is controlled by a resolution parameter; HDBSCAN is
density-based and infers the number of clusters on its own. Overlaying the labels on the
UMAP/t-SNE embeddings shows that the density-based and centroid-based methods disagree on
how the closely related endocrine populations are split.

---

## Classification

Cell types are predicted using supervised learning. Each model runs inside a scikit-learn
`Pipeline` (dimensionality reduction → classifier) so the PCA/SVD step is fit on training
folds only.

Algorithms evaluated:

- Random Forest
- Support Vector Machine
- Naive Bayes
- XGBoost 
- LightGBM 

Models are evaluated using **5-fold stratified cross-validation** and compared by
**accuracy** and **macro F1 score**.

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/janazivkovic02/Single-Cell-Pancreas-Analysis.git
cd Single-Cell-Pancreas-Analysis
```

2. Install the package and its dependencies

```bash
pip install -e .
```

This installs `src` as an importable package (so the notebook can simply do
`from src.config import ...` without any `sys.path` manipulation). Dependencies are declared
in `pyproject.toml`.

3. Add the data files to `data/raw/` (see the **Dataset** section above).

4. Run the analysis notebook

```bash
jupyter notebook main.ipynb
```
