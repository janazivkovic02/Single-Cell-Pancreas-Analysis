from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from mlxtend.frequent_patterns import fpgrowth, association_rules


@dataclass
class RulesConfig:
    n_top_genes: int = 100
    batch_key: Optional[str] = "batch"

    layer: Optional[str] = "counts"
    groupby: str = "assigned_cluster"
    min_frac_in_group: float = 0.10

    min_support: float = 0.10
    max_len: int = 2
    metric: str = "lift"
    min_threshold: float = 1.0


def _get_matrix(adata: sc.AnnData, layer: Optional[str]):
    """Return expression matrix from adata.layers[layer] or adata.X."""
    if layer is not None and layer in adata.layers:
        return adata.layers[layer]
    return adata.X


def select_hvgs_for_rules(
    adata: sc.AnnData,
    n_top: int = 100,
    batch_key: Optional[str] = "batch",
    key_added: str = "hvg_rules",
) -> np.ndarray:
    """
    Select highly variable genes for association rules analysis.
    """
    use_batch = batch_key if (batch_key is not None and batch_key in adata.obs.columns) else None

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top,
        flavor="seurat",
        batch_key=use_batch,
    )

    adata.var[key_added] = adata.var["highly_variable"].copy()
    return adata.var[key_added].to_numpy(dtype=bool)


def make_binary_df(
    adata: sc.AnnData,
    hvgs_mask: np.ndarray,
    *,
    layer: Optional[str] = "counts",
    groupby: str = "assigned_cluster",
    min_frac_in_group: float = 0.10,
) -> pd.DataFrame:
    """
    Build binary transaction matrix at group level.

    Rows = groups (e.g. clusters)
    Columns = selected genes
    Value = 1 if gene is expressed in at least min_frac_in_group fraction of cells in the group
    """
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby='{groupby}' not found in adata.obs.")

    genes = adata.var_names[hvgs_mask].to_list()
    if len(genes) == 0:
        raise ValueError("No genes selected for rules.")

    X_full = _get_matrix(adata, layer)
    X = X_full[:, hvgs_mask]

    if sparse.issparse(X):
        X_bin = X.copy()
        X_bin.data = np.ones_like(X_bin.data)
        X_bin = X_bin.astype(np.uint8)
    else:
        X_bin = (np.asarray(X) > 0).astype(np.uint8)

    groups = adata.obs[groupby].astype(str)
    unique_groups = groups.unique()

    rows = []
    row_names = []

    for g in unique_groups:
        mask = (groups.values == g)

        if sparse.issparse(X_bin):
            frac = np.asarray(X_bin[mask, :].mean(axis=0)).ravel()
        else:
            frac = X_bin[mask, :].mean(axis=0)

        row = (frac >= min_frac_in_group).astype(np.uint8)
        rows.append(row)
        row_names.append(g)

    return pd.DataFrame(np.vstack(rows), index=row_names, columns=genes, dtype=np.uint8)


def mine_fpgrowth_rules(
    binary_df: pd.DataFrame,
    *,
    min_support: float = 0.10,
    max_len: int = 2,
    metric: str = "lift",
    min_threshold: float = 1.0,
):
    """
    Run FP-Growth and derive association rules.
    """
    X = binary_df.astype(bool)

    frequent_itemsets = fpgrowth(
        X,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric=metric,
        min_threshold=min_threshold,
    )

    if not rules.empty:
        rules = rules.sort_values(
            ["lift", "confidence", "support"],
            ascending=False
        ).reset_index(drop=True)

    return frequent_itemsets, rules


def run_rules_pipeline(
    adata: sc.AnnData,
    cfg: RulesConfig = RulesConfig(),
) -> dict:
    """
    Full association-rules pipeline:
    1. select HVGs
    2. build binary group-level transaction matrix
    3. mine frequent itemsets and rules
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
    )

    frequent_itemsets, rules = mine_fpgrowth_rules(
        binary_df,
        min_support=cfg.min_support,
        max_len=cfg.max_len,
        metric=cfg.metric,
        min_threshold=cfg.min_threshold,
    )

    return {
        "hvgs_mask": hvgs_mask,
        "binary_df": binary_df,
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
    }


def rules_to_edge_list(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Convert single-gene association rules into an edge list.
    """
    if rules.empty:
        return pd.DataFrame(columns=["gene1", "gene2", "support", "confidence", "lift"])

    rows = []
    for _, r in rules.iterrows():
        ant = list(r["antecedents"])
        con = list(r["consequents"])

        if len(ant) == 1 and len(con) == 1:
            rows.append({
                "gene1": ant[0],
                "gene2": con[0],
                "support": float(r["support"]),
                "confidence": float(r["confidence"]),
                "lift": float(r["lift"]),
            })

    return pd.DataFrame(rows)