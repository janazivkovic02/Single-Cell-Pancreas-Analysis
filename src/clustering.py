# src/clustering.py
from __future__ import annotations

from typing import List, Dict
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


def cluster_kmeans(adata: sc.AnnData, k: int = 7, key: str = "kmeans_cluster", random_state: int = 42) -> None:
    X = adata.obsm["X_pca"]
    km = KMeans(n_clusters=k, random_state=random_state)
    adata.obs[key] = km.fit_predict(X).astype(str)


def cluster_leiden(adata: sc.AnnData, resolutions: List[float], rep_for_neighbors: str = "X_pca", n_neighbors: int = 15, key_prefix: str = "leiden_res", random_state: int = 42) -> None:
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=rep_for_neighbors)
    for r in resolutions:
        key = f"{key_prefix}{str(r).replace('.', '_')}"
        sc.tl.leiden(adata, key_added=key, resolution=r, random_state=random_state)


def cluster_spectral(adata: sc.AnnData, k: int = 7, key: str = "spectral_cluster", n_neighbors: int = 15, random_state: int = 42,assign_labels: str = "kmeans") -> None:
    X = adata.obsm["X_pca"]
    sp = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        random_state=random_state,
        assign_labels=assign_labels,
    )
    adata.obs[key] = sp.fit_predict(X).astype(str)


def cluster_hierarchical(adata: sc.AnnData, key: str = "hierarchical_cluster", n_clusters : int = 7,) -> None:
    X = adata.obsm["X_pca"]
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    adata.obs[key] = hc.fit_predict(X).astype(str)


def cluster_gmm(adata: sc.AnnData, k: int = 7, key: str = "gmm_cluster", random_state: int = 42) -> None:
    X = adata.obsm["X_pca"]
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
    gmm.fit(X)
    adata.obs[key] = gmm.predict(X).astype(str)


def silhouette_for_labels(adata: sc.AnnData, labels_key: str) -> float:
    X = adata.obsm["X_pca"]
    labels = adata.obs[labels_key].astype(str).to_numpy()
    return float(silhouette_score(X, labels))


def compare_clusterings(adata: sc.AnnData, label_keys: List[str]) -> pd.DataFrame:
    rows = []
    for k in label_keys:
        labels = pd.Categorical(adata.obs[k].astype(str))
        rows.append({"Model": k, "n_clusters": len(labels.categories), "Silhouette": silhouette_for_labels(adata, k)})
    return pd.DataFrame(rows).sort_values("Silhouette", ascending=False)
