from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict

from .config import RANDOM_STATE


def plot_gene_network(
    G_gene: nx.Graph,
    gene_to_module: Dict[str, int],
    largest_component: bool = True,
    seed: int = RANDOM_STATE,
    k: float = 0.45,
    figsize: tuple = (14, 10),
    title: str = "Gene network colored by detected gene modules",
) -> None:

    if G_gene.number_of_nodes() == 0:
        print("Graph is empty.")
        return

    if largest_component and G_gene.number_of_edges() > 0:
        largest_cc = max(nx.connected_components(G_gene), key=len)
        G_plot = G_gene.subgraph(largest_cc).copy()
    else:
        G_plot = G_gene.copy()

    pos = nx.spring_layout(G_plot, seed=seed, k=k)

    node_colors = [gene_to_module.get(node, -1) for node in G_plot.nodes()]

    degrees = dict(G_plot.degree())
    node_sizes = [300 + 180 * degrees[n] for n in G_plot.nodes()]

    edge_widths = []
    for u, v in G_plot.edges():
        lift = G_plot[u][v].get("lift", 1.0)
        edge_widths.append(1 + 1.5 * (lift - 1))

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        G_plot,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.tab20,
    )

    nx.draw_networkx_edges(
        G_plot,
        pos,
        width=edge_widths,
        alpha=0.6,
    )

    nx.draw_networkx_labels(
        G_plot,
        pos,
        font_size=9,
    )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()