"""
Visualize results for:
- tf
- kinase
- both

Reads:
- results/<tag>/network_edges.tsv
- results/<tag>/network_nodes.tsv (optional)
- data/differential/activities_early_<tag>.tsv
- data/differential/secretome_early_<tag>.tsv

Writes:
- results/<tag>/network.pdf + network.png
- results/<tag>/degree_distribution.pdf + degree_distribution.png
- results/<tag>/hubs.tsv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
FEATURE_TAGS = ["tf", "kinase", "both"]
SHOW_PLOTS = False


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
# Helpers
# -----------------------------------------------------------------------------
def build_network_graph(edges_df: pd.DataFrame, input_nodes: set[str], output_nodes: set[str]) -> graphviz.Digraph:
    g = graphviz.Digraph(
        engine="dot",
        graph_attr={"rankdir": "TB", "overlap": "false", "splines": "true", "fontname": "Helvetica"},
        node_attr={"shape": "box", "style": "rounded,filled", "fontname": "Helvetica", "fontsize": "10"},
        edge_attr={"fontname": "Helvetica", "fontsize": "8"},
    )

    if edges_df.empty:
        return g

    nodes = set(edges_df["source"]) | set(edges_df["target"])

    for n in sorted(nodes):
        if n == "TGFB1":
            g.node(n, fillcolor="#ff6b6b", fontcolor="white")
        elif n in input_nodes:
            g.node(n, fillcolor="#ffb3b3")
        elif n in output_nodes:
            g.node(n, fillcolor="#b3e6b3")
        else:
            g.node(n, fillcolor="#f0f0f0")

    for _, row in edges_df.iterrows():
        edge_color = "red" if float(row["edge_value"]) > 0 else "blue"
        arrowhead = "normal" if int(row["sign"]) > 0 else "tee"
        g.edge(str(row["source"]), str(row["target"]), color=edge_color, arrowhead=arrowhead)

    return g


def degree_plots(out_dir: Path, edges_df: pd.DataFrame) -> None:
    if edges_df.empty:
        return

    deg = {}
    for _, r in edges_df.iterrows():
        s = str(r["source"])
        t = str(r["target"])
        deg[s] = deg.get(s, 0) + 1
        deg[t] = deg.get(t, 0) + 1

    hubs = sorted(deg.items(), key=lambda x: x[1], reverse=True)
    hubs_df = pd.DataFrame(hubs, columns=["node", "degree"])
    hubs_df.to_csv(out_dir / "hubs.tsv", sep="\t", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.histplot(list(deg.values()), bins=20, ax=ax)
    ax.set_title("Degree distribution")
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_dir / "degree_distribution.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "degree_distribution.png", dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def visualize_tag(tag: str) -> None:
    out_dir = RESULTS_DIR / tag
    edges_path = out_dir / "network_edges.tsv"

    if not edges_path.exists():
        print(f"[skip] missing {edges_path}")
        return

    edges = pd.read_csv(edges_path, sep="\t")

    activities_early = pd.read_csv(DATA_DIR / "differential" / f"activities_early_{tag}.tsv", sep="\t")
    secretome_early = pd.read_csv(DATA_DIR / "differential" / f"secretome_early_{tag}.tsv", sep="\t")

    input_nodes = set(activities_early["source"]) | {"TGFB1"}
    output_nodes = set(secretome_early["id"])

    g = build_network_graph(edges, input_nodes, output_nodes)
    g.render(out_dir / "network", format="pdf", cleanup=True)
    g.render(out_dir / "network", format="png", cleanup=True)

    degree_plots(out_dir, edges)

    print(f"[done] {tag}: {len(edges)} edges -> {out_dir}")


if __name__ == "__main__":
    for tag in FEATURE_TAGS:
        visualize_tag(tag)