import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc


def plot_diffmap_per_cluster(
    adata,
    cluster_key="assigned_cluster",
    use_rep="X_pca",
    n_neighbors=10,
    min_cells=40,
    target_clusters=None,
):
    clusters = adata.obs[cluster_key].astype(str)

    if target_clusters is None:
        selected = sorted(clusters.unique())
    else:
        selected = [cl for cl in target_clusters if cl in set(clusters)]

    for cl in selected:
        mask = clusters == cl
        ad_sub = adata[mask].copy()

        if ad_sub.n_obs < min_cells:
            continue

        sc.pp.neighbors(
            ad_sub,
            n_neighbors=min(n_neighbors, ad_sub.n_obs - 1),
            use_rep=use_rep
        )
        sc.tl.diffmap(ad_sub)

        Xd = ad_sub.obsm["X_diffmap"]

        plt.figure(figsize=(5, 7))
        plt.scatter(Xd[:, 0], Xd[:, 1], s=8)
        plt.xlabel("DC1")
        plt.ylabel("DC2")
        plt.title(f"Diffusion map: cluster {cl} (n={ad_sub.n_obs})")
        plt.show()

def plot_diffmap_for_selected_clusters(
    adata,
    target_clusters,
    cluster_key="assigned_cluster",
    use_rep="X_pca",
    n_neighbors=15,
    figsize=(6, 5),
):
    clusters = adata.obs[cluster_key].astype(str)
    mask = clusters.isin(target_clusters)
    ad_sub = adata[mask].copy()

    sc.pp.neighbors(
        ad_sub,
        n_neighbors=min(n_neighbors, ad_sub.n_obs - 1),
        use_rep=use_rep
    )
    sc.tl.diffmap(ad_sub)

    Xd = ad_sub.obsm["X_diffmap"]
    labels = ad_sub.obs[cluster_key].astype(str)

    plt.figure(figsize=figsize)
    for cl in target_clusters:
        m = labels == cl
        plt.scatter(Xd[m, 0], Xd[m, 1], s=8, label=cl)

    plt.xlabel("DC1")
    plt.ylabel("DC2")
    plt.title(f"Diffusion map: {' / '.join(target_clusters)}")
    plt.legend()
    plt.show()

    plt.figure(figsize=(5, 7))
    plt.plot(ad_sub.uns["diffmap_evals"], "o-")
    plt.xlabel("component")
    plt.ylabel("eigenvalue")
    plt.title(f"Diffusion spectrum: {' / '.join(target_clusters)}")
    plt.show()

    return ad_sub

