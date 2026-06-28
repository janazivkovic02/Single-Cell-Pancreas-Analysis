from __future__ import annotations

from dataclasses import dataclass # Omogućava pravljenje “data-only” klasa bez ručnog pisanja __init__, npr. za RulesConfig sam dobila automatski generisan konstruktor
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from mlxtend.frequent_patterns import fpgrowth, association_rules

from .config import BATCH_KEY, CLUSTER_KEY, COUNTS_LAYER


@dataclass
class RulesConfig:
    n_top_genes: int = 100
    batch_key: Optional[str] = BATCH_KEY # Može biti ili ime ili None
    layer: Optional[str] = COUNTS_LAYER # Isto
    groupby: str = CLUSTER_KEY
    min_frac_in_group: float = 0.10
    min_support: float = 0.10
    max_len: int = 2
    metric: str = "lift"
    min_threshold: float = 1.0


def _get_matrix(adata: sc.AnnData, layer: Optional[str]):
    if layer is not None and layer in adata.layers:
        return adata.layers[layer]
    return adata.X


# Sada iz 4000 hvg biramo samo 100 koje cemo koristiti u izgradnji pravila pridruzivanja
def select_hvgs_for_rules(
    adata: sc.AnnData,
    n_top: int = 100,
    batch_key: Optional[str] = BATCH_KEY,
    key_added: str = "hvg_rules",
) -> np.ndarray:
    use_batch = batch_key if (batch_key is not None and batch_key in adata.obs.columns) else None

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top,
        flavor="seurat",
        batch_key=use_batch,
    )

    adata.var[key_added] = adata.var["highly_variable"].copy()
    return adata.var[key_added].to_numpy(dtype=bool)

# Pretvara scRNA-seq u market-basket format.
def make_binary_df(
    adata: sc.AnnData,
    hvgs_mask: np.ndarray,
    *,
    layer: Optional[str] = COUNTS_LAYER,
    groupby: str = CLUSTER_KEY,
    min_frac_in_group: float = 0.10,
) -> pd.DataFrame:
    # Grupisemo po assigned klasteru
    if groupby not in adata.obs.columns:
        raise ValueError(f"groupby='{groupby}' not found in adata.obs.")

    # Koristimo samo 100 gena
    genes = adata.var_names[hvgs_mask].to_list()
    if len(genes) == 0:
        raise ValueError("No genes selected for rules.")

    # Izdvajamo matricu vrednosti
    X = _get_matrix(adata, layer)[:, hvgs_mask]

    # Ako je vrednost u matrici veca od nule cuvam kao 1 inace ostavljam 0 
    if sparse.issparse(X):
        X_bin = X.copy()
        X_bin.data = np.ones_like(X_bin.data) # Sve nenulte vrednosti postaju 1 
        X_bin = X_bin.astype(np.uint8)
    else:
        X_bin = (np.asarray(X) > 0).astype(np.uint8)

    # Sada grupisem gene na osnovu klastera
    groups = adata.obs[groupby].astype(str)

    # Pravi se matrica koja za svoje redove ima ralzicite klastere, a kolone su geni i da li se oni pojavljuju u tom klasteru
    # Na osnovu ove baze se prave pravila pridruzivanja
    rows = []
    row_names = []
    for g in groups.unique():
        mask = groups.values == g
        if sparse.issparse(X_bin):
            frac = np.asarray(X_bin[mask, :].mean(axis=0)).ravel()
        else:
            frac = X_bin[mask, :].mean(axis=0)

        rows.append((frac >= min_frac_in_group).astype(np.uint8))
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
    frequent_itemsets = fpgrowth(
        binary_df.astype(bool),
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    if not rules.empty:
        rules = rules.sort_values(
            ["lift", "confidence", "support"], ascending=False
        ).reset_index(drop=True)

    return frequent_itemsets, rules


# Ova funkcija sluzi samo za pokretanje pipeline koji se odnosi na pravila pridruzvanja
def run_rules_pipeline(adata: sc.AnnData, cfg: RulesConfig = RulesConfig()) -> dict:
    hvgs_mask = select_hvgs_for_rules(
        adata, n_top=cfg.n_top_genes, batch_key=cfg.batch_key, key_added="hvg_rules"
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


def filter_rules(
    rules: pd.DataFrame,
    min_support: float = 0.10,
    min_confidence: float = 0.70,
    min_lift: float = 1.50,
) -> pd.DataFrame:
    # Zadrzavamo samo pravila koja su znacajna
    if rules.empty:
        return rules

    mask = (
        (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
        & (rules["lift"] >= min_lift)
    )
    return (
        rules[mask]
        .sort_values(["lift", "confidence", "support"], ascending=False)
        .reset_index(drop=True)
    )