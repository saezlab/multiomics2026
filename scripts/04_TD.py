"""
Network visualization and interpretation (TF-only / Kinase-only / Both).

Per tag:
- loads merged network results from results/<tag>/
- plots our network and the published early network in the same style
- prints edge/node overlap and degree distribution

Requires carnival_utils.load_results + carnival_utils.plot_network.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import corneto as cn
from corneto.graph import Graph

from carnival_utils import load_results, plot_network

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
try:
    _script_root = Path(__file__).resolve().parent.parent
except NameError:
    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results_new" if (_script_root / "data").is_dir() else Path("results_new")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TAGS = ["tf", "kinase", "both"]

# -----------------------------------------------------------------------------
# Load published network once
# -----------------------------------------------------------------------------
paper_edges = pd.read_csv(DATA_DIR / "network" / "paper_edges.tsv", sep="\t")
paper_nodes = pd.read_csv(DATA_DIR / "network" / "paper_nodes.tsv", sep="\t")

paper_early = paper_edges[paper_edges["network"] == "early"].copy()
paper_early_nodes = paper_nodes[paper_nodes["network"] == "early"].copy()

paper_early["sign_int"] = paper_early["sign"].str.strip("()").astype(int)

paper_edge_tuples = list(zip(paper_early["source"], paper_early["sign_int"], paper_early["target"]))
G_paper = Graph.from_tuples(paper_edge_tuples)

PAPER_ROLE_MAP = {
    "TF": "input",
    "Kinase/ phosphatase": "input",
    "Secreted proteins": "output",
}

paper_sample_data = {}
for _, row in paper_early_nodes.iterrows():
    role = PAPER_ROLE_MAP.get(row["type"])
    if role:
        paper_sample_data[row["node"]] = {"value": float(row["value"]), "mapping": "vertex", "role": role}

paper_data = cn.Data.from_cdict({"early": paper_sample_data})
paper_vertex_map = dict(zip(paper_early_nodes["node"], paper_early_nodes["value"]))
paper_vertex_values = [float(paper_vertex_map.get(name, 0.0)) for name in G_paper.V]
paper_edge_values = list(paper_early["sign_int"])

g_paper = G_paper.plot(
    preset="signaling",
    feature_data=paper_data,
    solution={"v": paper_vertex_values, "e": paper_edge_values},
    solution_map={"vertex": "v", "edge": "e"},
)

# -----------------------------------------------------------------------------
# Per tag visualization + comparison
# -----------------------------------------------------------------------------
for tag in TAGS:
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Our merged network
    edges, nodes = load_results("network", out_dir)
    print(f"\n[{tag}] Our network: {len(edges)} edges, {len(nodes)} nodes")

    g = plot_network(edges, nodes)
    g.render(out_dir / "network", format="pdf", cleanup=True)
    g.render(out_dir / "network", format="png", cleanup=True)

    # Paper network (same in each folder for convenience)
    g_paper.render(out_dir / "network_paper_early", format="pdf", cleanup=True)
    g_paper.render(out_dir / "network_paper_early", format="png", cleanup=True)

    # Edge overlap
    paper_pairs = set(zip(paper_early["source"], paper_early["target"]))
    our_pairs = set(zip(edges["source"], edges["target"])) if len(edges) else set()

    overlap = paper_pairs & our_pairs
    only_paper = paper_pairs - our_pairs
    only_ours = our_pairs - paper_pairs

    print(f"[{tag}] Edge comparison vs published early:")
    print(f"  Shared edges: {len(overlap)}")
    print(f"  Only in published: {len(only_paper)}")
    print(f"  Only in ours: {len(only_ours)}")

    if len(paper_pairs | our_pairs) > 0:
        jaccard = len(overlap) / len(paper_pairs | our_pairs)
        print(f"  Jaccard similarity: {jaccard:.3f}")

    # Node overlap
    paper_node_set = set(paper_early_nodes["node"])
    our_node_set = set(nodes["node"]) if len(nodes) else set()
    node_overlap = paper_node_set & our_node_set

    print(f"[{tag}] Node comparison:")
    print(f"  Shared nodes: {len(node_overlap)}")
    print(f"  Only in published: {len(paper_node_set - our_node_set)}")
    print(f"  Only in ours: {len(our_node_set - paper_node_set)}")

    # Degree distribution
    if len(edges) > 0:
        degree = pd.concat([
            edges["source"].value_counts().rename("out_degree"),
            edges["target"].value_counts().rename("in_degree"),
        ], axis=1).fillna(0).astype(int)
        degree["total_degree"] = degree["out_degree"] + degree["in_degree"]
        degree = degree.sort_values("total_degree", ascending=False)

        # Save hubs
        degree.to_csv(out_dir / "degree_table.tsv", sep="\t")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(degree["total_degree"], bins=20, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Node degree")
        ax.set_ylabel("Count")
        ax.set_title(f"Degree distribution ({tag})")
        plt.tight_layout()
        plt.savefig(out_dir / "degree_distribution.pdf", bbox_inches="tight")
        plt.savefig(out_dir / "degree_distribution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)