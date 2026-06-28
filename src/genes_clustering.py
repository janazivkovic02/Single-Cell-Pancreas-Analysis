from __future__ import annotations

import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

# Ova funkcija sluzi da konstruisem graf
def rules_to_gene_network(rules_f: pd.DataFrame) -> nx.Graph:
    # Krecem od praznog grafa
    G = nx.Graph()
    if rules_f.empty:
        return G

    edges = []
    for _, r in rules_f.iterrows():
        ant = list(r["antecedents"])
        con = list(r["consequents"])
        if len(ant) == 1 and len(con) == 1: # Filtriraju se samo 1-1 pravila (A -> B je jasna veza i moze da se protumaci u smislu grafa)
            a, b = sorted([ant[0], con[0]]) # Sortiramo da se ne bi ponavljala pravila dva puta (A -> B i B -> A tj da bi ovo bila ista ivica)
            # Ideja je sledeca
            # A --- B
            # weight = lift (veći lift → jača veza → više šanse da budu u istom modulu)
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

    edges_df = (
        pd.DataFrame(edges)
        .sort_values(["lift", "confidence", "support"], ascending=False)
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
    # Modulariti meri su čvorovi gusti unutar grupa u odnosu na slučajni graf
    communities = greedy_modularity_communities(G, weight="weight")
    return [sorted(list(c)) for c in communities if len(c) >= min_size] # Takodje zelimo da odbacimo male modelu tj. ako se klasteruje samo par gena


def modules_to_gene_map(modules: list[list[str]]) -> dict[str, int]:
    return {gene: i for i, module in enumerate(modules) for gene in module}


def modules_to_df(modules: list[list[str]]) -> pd.DataFrame:
    rows = [
        {"module_id": i, "n_genes": len(module), "genes": ", ".join(module)}
        for i, module in enumerate(modules)
    ]
    return pd.DataFrame(rows)


# Vracamo graf u dataframe radi vizualizacije 
def gene_network_edges_df(G: nx.Graph) -> pd.DataFrame:
    rows = [
        {
            "gene1": u,
            "gene2": v,
            "support": d.get("support"),
            "confidence": d.get("confidence"),
            "lift": d.get("lift"),
        }
        for u, v, d in G.edges(data=True)
    ]
    return pd.DataFrame(rows)