#učitavanje CSV→AnnData, concat

# src/io.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import pandas as pd
import scanpy as sc

def csv_to_adata(
    path: str | Path,
    metadata_cols: List[str] = ["Unnamed: 0", "barcode", "assigned_cluster"],
) -> sc.AnnData:
    df = pd.read_csv(path, compression="gzip")
    gene_cols = df.columns.difference(metadata_cols)

    X = df[gene_cols].values
    adata = sc.AnnData(X)

    adata.obs = df[metadata_cols].copy()
    adata.obs_names = df["barcode"].astype(str)
    adata.var_names = gene_cols

    return adata