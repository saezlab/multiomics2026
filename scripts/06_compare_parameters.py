import pandas as pd
from pathlib import Path
import igraph as ig
import matplotlib.pyplot as plt
from upsetplot import from_contents
from upsetplot import UpSet
from collections import defaultdict
import seaborn as sns

from carnival_utils import (
    run_carnival, extract_results, plot_network,
    save_results, merge_networks, print_summary,
)

try:
    _script_root = Path(__file__).resolve().parent.parent
except NameError:
    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results" if (_script_root / "data").is_dir() else Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

lambda_values = [0.1, 0.01]
solver = ["HIGHS", "CVXOPT", "GLPK", "GLPK_MI"] #["CVXOPT", "GLPK", "GLPK_MI", "HIGHS", "SCIP", "SCIPY"]
time_limit = [300,500]

stats = {}
for lval in lambda_values:
    for solv in solver:
        for timel in time_limit:

            run_name = f"{lval}_{solv}_{timel}"
            stat = {}
            network = pd.read_csv(RESULTS_DIR / f"network_merged_{lval}_{solv}_{timel}_edges.tsv", sep="\t")
            node_network = pd.read_csv(RESULTS_DIR / f"network_merged_{lval}_{solv}_{timel}_nodes.tsv", sep="\t")

            G = ig.Graph.DataFrame(
                network[["source", "target", "sign", "edge_value"]],
                directed=True,
                vertices=node_network[["node", "value", "type"]],
                use_vids=False,
            )
            sources = network["source"].to_list()
            targets = network["target"].to_list()
            stat["edgelist"] = [(sources[x],targets[x]) for x in range(0, len(sources))] # G.get_edgelist() # now it is wiht numbers and not with names!!!
            stat["verticlist"] = node_network["node"].to_list()
            stat["vertices"] = G.vcount()
            stat["edges"] = G.ecount()
            stat["components"] = len(G.connected_components(mode='weak'))
            stat["betweenness"] = G.betweenness(directed=True)
            stat["page_rank"] = G.pagerank(directed=True)
            stat["degree_in"] = G.degree(mode="in")
            stat["degree_out"] = G.degree(mode="out")
            stat["degree_total"] = G.degree()

            print(stat)
            stats[run_name] = stat

def boxplot_groups(graph_stats, stat):
    # Group graph names by prefix (or any logic)
    groups = defaultdict(list)
    for graph_name in graph_stats:
        group_name = graph_name.split("_")[1]
        groups[group_name].append(graph_name)
    
    # Sort groups and graphs within groups
    sorted_groups = sorted(groups.keys())
    
    data = []
    labels = []
    for group in sorted_groups:
        for graph_name in sorted(groups[group]):
            data.append(graph_stats[graph_name][stat])
            labels.append(graph_name)

    # Plot
    plt.figure(figsize=(12,6))
    plt.boxplot(data, tick_labels=labels)
    plt.xlabel("Graph")
    plt.ylabel(stat)
    plt.title(f"{stat} Distribution per Graph (Grouped)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    plt.savefig(RESULTS_DIR / f"{stat}_boxplot_grouped.png", dpi=300)
    #plt.show()

boxplot_groups(stats, "betweenness")
boxplot_groups(stats, "degree_in")
boxplot_groups(stats, "degree_out")
boxplot_groups(stats, "degree_total")
boxplot_groups(stats, "page_rank")

def bar_plot(graph_stats, stat):
    labels = list(graph_stats.keys())
    data = [graph_stats[g][stat] for g in labels]

    plt.figure(figsize=(10,6))
    plt.bar(labels, data)

    plt.xlabel("Graph")
    plt.ylabel(f"Number of {stat}")
    plt.title(f"{stat} per Graph")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(RESULTS_DIR /f"{stat}_per_graph.png", dpi=300)
    #plt.show()

bar_plot(stats, "edges")
bar_plot(stats, "vertices")
bar_plot(stats, "components")

def heatmap_plot(stats):
    graph_names = list(stats.keys())
    edges_lists = [stats[g]["edgelist"] for g in graph_names]

    # Flatten all edges to get unique set
    all_edges = set()
    for elist in edges_lists:
        all_edges.update(elist)
    all_edges = sorted(all_edges)  # sort for consistent ordering

    # Build binary matrix: rows=edges, columns=graphs
    heatmap_data = pd.DataFrame(
        0, 
        index=[f"{e[0]}-{e[1]}" for e in all_edges],  # edge as "node1-node2"
        columns=graph_names
    )

    for col, edges in zip(graph_names, edges_lists):
        for edge in edges:
            heatmap_data.loc[f"{edge[0]}-{edge[1]}", col] = 1

    # Plot clustermap
    sns.clustermap(
        heatmap_data,
        annot=False,
        cmap="RdBu_r",
        linewidths=0.5,
        figsize=(10, 50),
        metric="euclidean",
        method="average",
        dendrogram_ratio=(0.1, 0.02),
        cbar_pos=(0.95, 0.2, 0.03, 0.4),
    )
    plt.savefig(RESULTS_DIR /f"heatmap", dpi=300)

heatmap_plot(stats)
