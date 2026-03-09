# Single-Cell Pancreas Analysis  
**Classification, Clustering and Association Rule Mining on Single-Cell RNA-seq Data**

## Overview

This project analyzes single-cell RNA sequencing (scRNA-seq) data from the **Human Pancreas Single-Cell Transcriptomic Atlas** using several data mining and machine learning techniques.

The main goal of the project is to explore the structure of the data and evaluate different approaches for:

- **Association rule mining** on gene expression  
- **Clustering of cells**  
- **Classification of cell types**  
- **Geometric analysis of the data manifold**

The analysis follows a standard scRNA-seq preprocessing pipeline and then applies several machine learning methods to understand both the **biological structure of the dataset** and the **behavior of different algorithms**.

An additional focus of the project is understanding how the **geometric structure of the data influences clustering results**, particularly the difference between centroid-based clustering methods and density-based clustering methods.

---

# Dataset

The analysis uses the dataset:

**A Single-Cell Transcriptomic Atlas of the Human and Mouse Pancreas**

The data contains UMI counts for thousands of genes measured in individual pancreatic cells.

Files used in this project:

GSM2230757_human1_umifm_counts.csv.gz
GSM2230758_human2_umifm_counts.csv.gz
GSM2230759_human3_umifm_counts.csv.gz
GSM2230760_human4_umifm_counts.csv.gz

# Project Structure

Single-Cell-Pancreas-Analysis

│

├── data/

│ └── raw/

│

├── notebooks/

│ ├── main.ipynb

│ ├── geometric_analysis.ipynb

│ └── outputs/

│ ├── figures/

│ ├── tables/

│ └── adata/

│

├── src/

│ ├── data_loader.py

│ ├── quality_control.py

│ ├── preprocess.py

│ ├── rules.py

│ ├── genes_clustering.py

│ ├── cells_clustering.py

│ ├── classification.py

│ └── vis.py

│

└── README.md


### notebooks/

**main.ipynb**

Main analysis pipeline including:

- preprocessing
- association rule mining
- gene network construction
- clustering comparison
- classification experiments

**geometric_analysis.ipynb**

Additional analysis of the geometric structure of the dataset including:

- diffusion maps
- intrinsic dimensionality
- investigation of endocrine cell manifold structure

---

# Data Preprocessing

The preprocessing pipeline follows standard scRNA-seq analysis practices.

Steps include:

1. Loading raw UMI count matrices  
2. Quality control and outlier detection  
3. Normalization and log transformation  
4. Selection of highly variable genes (HVG)  
5. Dimensionality reduction using PCA  

The processed dataset is stored as an **AnnData object** for efficient downstream analysis.

---

# Association Rule Mining

Association rules are applied to gene expression patterns to identify **co-expressed genes**.

Pipeline:

1. Selection of highly variable genes  
2. Binarization of gene expression  
3. Frequent itemset mining  
4. Rule filtering using support, confidence and lift  
5. Construction of a **gene association network**

Gene modules are detected by clustering the resulting gene network.

This approach reveals groups of genes with similar expression patterns across cell populations.

---

# Cell Clustering

Several clustering algorithms are evaluated on the PCA representation of the dataset.

Algorithms used:

- **KMeans**
- **Gaussian Mixture Models (GMM)**
- **Spectral Clustering**
- **HDBSCAN**
- **Leiden community detection**

Clustering results are compared using the **Silhouette score**.

The analysis shows that density-based clustering can capture the global geometry of the data manifold more effectively than centroid-based methods.

---

# Classification

Cell types are predicted using supervised learning models.

Algorithms evaluated:

- Random Forest
- Support Vector Machines
- Naive Bayes
- XGBoost
- LightGBM

Models are evaluated using **5-fold cross-validation** on PCA features.

Performance metrics include:

- accuracy
- macro F1 score

---

# Geometric Analysis of the Data

To better understand clustering results, additional analysis of the data geometry was performed.

Methods include:

- **Diffusion maps**
- **Intrinsic dimensionality estimation**
- **Local PCA spectra**

This analysis revealed that some endocrine cell populations (alpha, beta and delta cells) occupy nearby regions of a shared low-dimensional manifold.

This explains why **density-based clustering methods such as HDBSCAN may merge these populations**, while centroid-based methods separate them.

---
