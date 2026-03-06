# src/genes_clustering.py
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def rules_to_gene_network(rules_f: pd.DataFrame, directed: bool = False):
    G = nx.DiGraph() if directed else nx.Graph()

    if rules_f.empty:
        return G

    if directed:
        for _, r in rules_f.iterrows():
            ant = list(r["antecedents"])
            con = list(r["consequents"])

            if len(ant) == 1 and len(con) == 1:
                g1, g2 = ant[0], con[0]
                G.add_edge(
                    g1,
                    g2,
                    support=float(r["support"]),
                    confidence=float(r["confidence"]),
                    lift=float(r["lift"]),
                    weight=float(r["lift"]),
                )
        return G

    # undirected version: merge A-B and B-A
    edges = []
    for _, r in rules_f.iterrows():
        ant = list(r["antecedents"])
        con = list(r["consequents"])

        if len(ant) == 1 and len(con) == 1:
            g1, g2 = ant[0], con[0]
            a, b = sorted([g1, g2])
            edges.append(
                {
                    "gene1": a,
                    "gene2": b,
                    "support": float(r["support"]),
                    "confidence": float(r["confidence"]),
                    "lift": float(r["lift"]),
                }
            )

    if not edges:
        return G

    edges_df = pd.DataFrame(edges)
    edges_df = (
        edges_df.sort_values(["lift", "confidence", "support"], ascending=False)
        .drop_duplicates(subset=["gene1", "gene2"])
        .reset_index(drop=True)
    )

    for _, r in edges_df.iterrows():
        G.add_edge(
            r["gene1"],
            r["gene2"],
            support=float(r["support"]),
            confidence=float(r["confidence"]),
            lift=float(r["lift"]),
            weight=float(r["lift"]),
        )

    return G


def cluster_genes_from_network(G: nx.Graph, min_size: int = 2) -> list[list[str]]:
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return []

    communities = greedy_modularity_communities(G, weight="weight")
    modules = [sorted(list(c)) for c in communities if len(c) >= min_size]
    return modules


def modules_to_df(modules: list[list[str]]) -> pd.DataFrame:
    rows = []
    for i, module in enumerate(modules):
        rows.append(
            {
                "module_id": i,
                "n_genes": len(module),
                "genes": ", ".join(module),
            }
        )
    return pd.DataFrame(rows)


def gene_network_edges_df(G) -> pd.DataFrame:
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append(
            {
                "gene1": u,
                "gene2": v,
                "support": d.get("support"),
                "confidence": d.get("confidence"),
                "lift": d.get("lift"),
            }
        )
    return pd.DataFrame(rows)

