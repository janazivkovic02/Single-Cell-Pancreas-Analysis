# src/qc.py
from __future__ import annotations

import numpy as np
import scanpy as sc
from scipy.stats import median_abs_deviation


def annotate_qc_genes(adata: sc.AnnData) -> None:
    """Označi mt/ribo/hb gene u adata.var."""
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]", regex=True)


def compute_qc_metrics(adata: sc.AnnData) -> None:
    """Scanpy QC metrics (isto kao u tvom kodu)."""
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        inplace=True,
        percent_top=[20],
        log1p=True,
    )


def is_outlier_mad(adata: sc.AnnData, metric: str, nmads: int = 5) -> np.ndarray:
    """
    Flaguje outlier-e po MAD pravilu koje koristiš.
    Vrati boolean masku dužine n_obs.
    """
    M = adata.obs[metric].to_numpy()
    med = np.median(M)
    mad = median_abs_deviation(M)
    if mad == 0:
        # ako je mad 0, nema smislenog outlier pravila; vrati sve False
        return np.zeros_like(M, dtype=bool)

    low = med - nmads * mad
    high = med + nmads * mad
    return (M < low) | (M > high)


def flag_outliers(adata: sc.AnnData, nmads: int = 5, key: str = "outlier") -> None:
    """
    Kreira adata.obs[key] po tvom OR pravilu preko 3 metrike.
    """
    out = (
        is_outlier_mad(adata, "log1p_total_counts", nmads)
        | is_outlier_mad(adata, "log1p_n_genes_by_counts", nmads)
        | is_outlier_mad(adata, "pct_counts_in_top_20_genes", nmads)
    )
    adata.obs[key] = out
