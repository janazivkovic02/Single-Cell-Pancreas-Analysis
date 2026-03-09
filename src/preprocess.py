# src/preprocess.py

from __future__ import annotations

import scanpy as sc
import numpy as np


def store_raw_counts(adata: sc.AnnData) -> None:
    adata.layers["counts"] = adata.X.copy()


def normalize_and_log(adata: sc.AnnData, target_sum: float = 1e4) -> None:
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()


def select_hvgs(
    adata: sc.AnnData,
    store_key: str = "hvg_flag",
) -> None:
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=4000,
        flavor="seurat",
        batch_key="batch",
    )
    adata.var[store_key] = adata.var["highly_variable"].copy()

