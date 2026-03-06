# src/rules.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import chi2_contingency

from mlxtend.frequent_patterns import fpgrowth, association_rules


BinarizeMode = Literal["gt0", "quantile"]


@dataclass
class RulesConfig:
    # Feature selection
    n_top_genes: int = 100
    batch_key: Optional[str] = "batch"  # set None if you don't want batch-aware HVG

    # Transactions building
    layer: Optional[str] = "counts"     # use "counts" if you stored raw counts, else None for adata.X
    groupby: Optional[str] = "assigned_cluster"  # e.g. "assigned_cluster", "leiden", "sample", "batch"
    min_frac_in_group: float = 0.10     # gene is "present" in a group if expressed in >= this fraction of cells

    # If groupby is None -> cell-level transactions (use with caution)
    max_cells: int = 20000
    random_state: int = 42

    # Binarization
    mode: BinarizeMode = "gt0"
    quantile: float = 0.90  # used only if mode == "quantile"

    # Mining
    min_support: float = 0.10
    max_len: int = 2
    metric: str = "lift"
    min_threshold: float = 1.0

    # Optional statistical filtering (post-hoc)
    apply_chi2_filter: bool = False
    chi2_alpha: float = 0.05


def _get_matrix(adata: sc.AnnData, layer: Optional[str]) -> sparse.spmatrix | np.ndarray:
    """Return expression matrix from adata.X or adata.layers[layer]."""
    X = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
    return X


def select_hvgs_for_rules(
    adata: sc.AnnData,
    n_top: int = 100,
    batch_key: Optional[str] = "batch",
    flavor: str = "seurat",
    key_added: str = "hvg_rules",
) -> np.ndarray:
    """
    Select HVGs intended for association rules, store boolean mask in adata.var[key_added],
    and return the mask.

    Notes:
    - For rules, you typically want a small n_top (e.g., 50-300) to keep search space sane.
    - If batch_key is provided and exists in adata.obs, selection is batch-aware.
    """
    use_batch = batch_key if (batch_key is not None and batch_key in adata.obs.columns) else None
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top,
        flavor=flavor,
        batch_key=use_batch,
    )
    adata.var[key_added] = adata.var["highly_variable"].copy()
    return adata.var[key_added].to_numpy(dtype=bool)


def _binarize_cell_level(
    X: sparse.spmatrix | np.ndarray,
    mode: BinarizeMode = "gt0",
    quantile: float = 0.90,
) -> sparse.spmatrix | np.ndarray:
    """
    Binarize matrix at cell-level.
    - gt0: 1 if expression > 0
    - quantile: per-gene threshold at given quantile, 1 if >= threshold
    """
    if mode == "gt0":
        if sparse.issparse(X):
            X_bin = X.copy()
            X_bin.data = np.ones_like(X_bin.data)
            return X_bin
        return (X > 0).astype(np.uint8)

    if mode == "quantile":
        # compute per-gene thresholds
        if sparse.issparse(X):
            # sparse quantile per column is expensive; convert to dense only if small enough
            X_dense = X.toarray()
            thr = np.quantile(X_dense, quantile, axis=0)
            return (X_dense >= thr).astype(np.uint8)
        thr = np.quantile(X, quantile, axis=0)
        return (X >= thr).astype(np.uint8)

    raise ValueError(f"Unknown mode: {mode}")


def make_binary_df(
    adata: sc.AnnData,
    hvgs_mask: np.ndarray,
    *,
    layer: Optional[str] = "counts",
    groupby: Optional[str] = "assigned_cluster",
    min_frac_in_group: float = 0.10,
    max_cells: int = 20000,
    random_state: int = 42,
    mode: BinarizeMode = "gt0",
    quantile: float = 0.90,
) -> pd.DataFrame:
    """
    Build a binary transaction DataFrame (rows=transactions, cols=genes).

    Two modes:
    1) groupby != None: transactions are groups (recommended for scRNA).
       A gene is present in a group if expressed in >= min_frac_in_group of cells in that group.
    2) groupby == None: transactions are cells; optionally subsample cells to max_cells.
       Use with caution on big datasets.

    Returns:
        pd.DataFrame of shape (n_transactions, n_genes_hvg) with 0/1 values.
    """
    genes = adata.var_names[hvgs_mask].to_list()
    if len(genes) == 0:
        raise ValueError("hvgs_mask selected 0 genes.")

    X_full = _get_matrix(adata, layer)
    # subset genes
    if sparse.issparse(X_full):
        X = X_full[:, hvgs_mask]
    else:
        X = np.asarray(X_full)[:, hvgs_mask]

    # CELL-LEVEL TRANSACTIONS
    if groupby is None:
        n = adata.n_obs
        rng = np.random.default_rng(random_state)
        if n > max_cells:
            idx = rng.choice(n, size=max_cells, replace=False)
            X = X[idx, :]
            index = adata.obs_names[idx]
        else:
            index = adata.obs_names

        X_bin = _binarize_cell_level(X, mode=mode, quantile=quantile)
        if sparse.issparse(X_bin):
            df = pd.DataFrame.sparse.from_spmatrix(X_bin, index=index, columns=genes)
            # ensure 0/1 integers in sparse dtype
            return df.astype(pd.SparseDtype("uint8", fill_value=0))
        return pd.DataFrame(X_bin, index=index, columns=genes, dtype=np.uint8)

    # GROUP-LEVEL TRANSACTIONS (PSEUDO-BULK BINARY)
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby='{groupby}' not found in adata.obs columns.")

    groups = adata.obs[groupby].astype(str)
    uniq = groups.unique()

    # For each group, compute fraction of cells with gene > 0
    # Efficiently: for sparse, use (X > 0) as bool sparse then sum per group.
    if sparse.issparse(X):
        X_bool = X.copy()
        X_bool.data = np.ones_like(X_bool.data)  # now counts presence
        X_bool = X_bool.astype(np.uint8)
    else:
        X_bool = (X > 0).astype(np.uint8)

    out = []
    out_index = []
    for g in uniq:
        mask = (groups.values == g)
        ng = int(mask.sum())
        if ng == 0:
            continue

        if sparse.issparse(X_bool):
            frac = np.asarray(X_bool[mask, :].mean(axis=0)).ravel()
        else:
            frac = X_bool[mask, :].mean(axis=0)

        row = (frac >= min_frac_in_group).astype(np.uint8)
        out.append(row)
        out_index.append(g)

    df = pd.DataFrame(np.vstack(out), index=out_index, columns=genes, dtype=np.uint8)
    return df


def mine_fpgrowth_rules(
    binary_df: pd.DataFrame,
    *,
    min_support: float = 0.10,
    max_len: int = 2,
    metric: str = "lift",
    min_threshold: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run FP-Growth and derive association rules.

    Returns:
        frequent_itemsets, rules
    """
    # mlxtend expects boolean or 0/1
    X = binary_df.astype(bool)

    frequent = fpgrowth(X, min_support=min_support, use_colnames=True, max_len=max_len)
    if frequent.empty:
        rules = pd.DataFrame()
        return frequent, rules

    rules = association_rules(frequent, metric=metric, min_threshold=min_threshold)
    # Sort for convenience
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence", "support"], ascending=False).reset_index(drop=True)
    return frequent, rules


def chi2_filter_rules(
    binary_df: pd.DataFrame,
    rules: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Optional post-hoc filter: keep only rules where antecedent and consequent are dependent by chi-square test.

    This is conservative and mostly useful when you have many transactions (not too small groupby).
    """
    if rules.empty:
        return rules

    X = binary_df.astype(np.uint8).to_numpy()
    cols = list(binary_df.columns)
    col_idx = {c: i for i, c in enumerate(cols)}

    keep = []
    pvals = []
    for i, r in rules.iterrows():
        # We assume max_len=2 (single antecedent + single consequent) for interpretability.
        ant = list(r["antecedents"])
        con = list(r["consequents"])
        if len(ant) != 1 or len(con) != 1:
            keep.append(False)
            pvals.append(np.nan)
            continue

        a = col_idx[ant[0]]
        c = col_idx[con[0]]

        xa = X[:, a]
        xc = X[:, c]

        # contingency table: [a0/a1] x [c0/c1]
        n11 = int(np.sum((xa == 1) & (xc == 1)))
        n10 = int(np.sum((xa == 1) & (xc == 0)))
        n01 = int(np.sum((xa == 0) & (xc == 1)))
        n00 = int(np.sum((xa == 0) & (xc == 0)))

        table = np.array([[n00, n01], [n10, n11]], dtype=np.int64)
        try:
            _, p, _, _ = chi2_contingency(table, correction=False)
        except Exception:
            p = np.nan

        pvals.append(p)
        keep.append(bool(p < alpha) if not np.isnan(p) else False)

    out = rules.copy()
    out["chi2_pvalue"] = pvals
    out = out.loc[keep].reset_index(drop=True)
    return out


def run_rules_pipeline(
    adata: sc.AnnData,
    cfg: RulesConfig = RulesConfig(),
) -> dict:
    """
    One-call pipeline:
    1) select HVGs
    2) build binary transactions
    3) mine FP-Growth itemsets + rules
    4) optional chi-square filtering

    Returns dict with keys: hvgs_mask, binary_df, frequent_itemsets, rules
    """
    hvgs_mask = select_hvgs_for_rules(
        adata,
        n_top=cfg.n_top_genes,
        batch_key=cfg.batch_key,
        key_added="hvg_rules",
    )

    binary_df = make_binary_df(
        adata,
        hvgs_mask,
        layer=cfg.layer,
        groupby=cfg.groupby,
        min_frac_in_group=cfg.min_frac_in_group,
        max_cells=cfg.max_cells,
        random_state=cfg.random_state,
        mode=cfg.mode,
        quantile=cfg.quantile,
    )

    frequent, rules = mine_fpgrowth_rules(
        binary_df,
        min_support=cfg.min_support,
        max_len=cfg.max_len,
        metric=cfg.metric,
        min_threshold=cfg.min_threshold,
    )

    if cfg.apply_chi2_filter and not rules.empty:
        rules = chi2_filter_rules(binary_df, rules, alpha=cfg.chi2_alpha)

    return {
        "hvgs_mask": hvgs_mask,
        "binary_df": binary_df,
        "frequent_itemsets": frequent,
        "rules": rules,
    }


def rules_to_edge_list(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Convert (single antecedent -> single consequent) rules into a network edge list.

    Output columns:
        gene1, gene2, support, confidence, lift
    """
    if rules.empty:
        return pd.DataFrame(columns=["gene1", "gene2", "support", "confidence", "lift"])

    rows = []
    for _, r in rules.iterrows():
        ant = list(r["antecedents"])
        con = list(r["consequents"])
        if len(ant) == 1 and len(con) == 1:
            rows.append(
                {
                    "gene1": ant[0],
                    "gene2": con[0],
                    "support": float(r["support"]),
                    "confidence": float(r["confidence"]),
                    "lift": float(r["lift"]),
                }
            )
    return pd.DataFrame(rows)