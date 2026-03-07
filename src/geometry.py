# src/geometric.py
from __future__ import annotations

import scanpy as sc
import pandas as pd
import numpy as np

def compute_diffusion_map(
    adata: sc.AnnData,
    rep_key: str = "X_pca",
    n_neighbors: int = 15,
    n_comps: int = 10,
    random_state: int = 42,
) -> None:
    """
    Računa neighbors graf i diffusion map embedding.
    Rezultat se čuva u:
    - adata.obsm["X_diffmap"]
    - adata.uns["diffmap_evals"]
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep_key, random_state=random_state)
    sc.tl.diffmap(adata, n_comps=n_comps)

def diffusion_components_df(
    adata: sc.AnnData,
    n_components: int = 5,
    prefix: str = "DC",
) -> pd.DataFrame:
    X = adata.obsm["X_diffmap"][:, :n_components]
    cols = [f"{prefix}{i+1}" for i in range(n_components)]
    return pd.DataFrame(X, index=adata.obs_names, columns=cols)

def compute_dpt(
    adata: sc.AnnData,
    root_index: int,
) -> None:
    """
    Računa diffusion pseudotime.
    root_index je indeks početne ćelije.
    """
    adata.uns["iroot"] = root_index
    sc.tl.dpt(adata)