from __future__ import annotations # STA OVO ZNACI

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import median_abs_deviation

from .config import BARCODE_KEY, BATCH_KEY, CLUSTER_KEY, COUNTS_LAYER

# Default metadata columns kept as a tuple (immutable default argument).
DEFAULT_METADATA_COLS: tuple[str, ...] = ("Unnamed: 0", BARCODE_KEY, CLUSTER_KEY)

# Učitavanje
def csv_to_adata(
    path: str | Path,
    metadata_cols: Sequence[str] = DEFAULT_METADATA_COLS,
) -> sc.AnnData:
    metadata_cols = list(metadata_cols)
    df = pd.read_csv(path, compression="gzip")
    gene_cols = df.columns.difference(metadata_cols)

    adata = sc.AnnData(df[gene_cols].values)
    adata.obs = df[metadata_cols].copy()
    adata.obs_names = df[BARCODE_KEY].astype(str)
    adata.var_names = gene_cols
    return adata


# Analiza kontrole (ovo je standardno u ovakvim zadacima)
def annotate_qc_genes(adata: sc.AnnData) -> None:
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]", regex=True)


def compute_qc_metrics(adata: sc.AnnData) -> None:
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=[20],
        log1p=True,
    )


def is_outlier_mad(adata: sc.AnnData, metric: str, nmads: int = 5) -> np.ndarray:
    M = adata.obs[metric].to_numpy()
    med = np.median(M)
    mad = median_abs_deviation(M)
    if mad == 0:
        # No spread -> no meaningful outlier rule; treat nothing as an outlier.
        return np.zeros_like(M, dtype=bool)

    low = med - nmads * mad
    high = med + nmads * mad
    return (M < low) | (M > high)


def flag_outliers(adata: sc.AnnData, nmads: int = 5, key: str = "outlier") -> None:
    out = (
        is_outlier_mad(adata, "log1p_total_counts", nmads)
        | is_outlier_mad(adata, "log1p_n_genes_by_counts", nmads)
        | is_outlier_mad(adata, "pct_counts_in_top_20_genes", nmads)
    )
    adata.obs[key] = out


# Preprocesiranje
def store_raw_counts(adata: sc.AnnData, layer: str = COUNTS_LAYER) -> None:
    adata.layers[layer] = adata.X.copy()


def normalize_and_log(adata: sc.AnnData, target_sum: float = 1e4) -> None:
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()


def select_hvgs(
    adata: sc.AnnData,
    n_top_genes: int = 4000,
    batch_key: str | None = BATCH_KEY,
    store_key: str = "hvg_flag",
) -> None:
    use_batch = batch_key if (batch_key is not None and batch_key in adata.obs.columns) else None
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat",
        batch_key=use_batch,
    )
    adata.var[store_key] = adata.var["highly_variable"].copy()