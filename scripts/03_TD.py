"""
Run CORNETO/CARNIVAL for each tag:
- tf
- kinase
- both

Reads:
- data/network/pkn_<tag>.tsv
- data/differential/activities_early_<tag>.tsv
- data/differential/secretome_early_<tag>.tsv

Writes:
- results/<tag>/network_corneto.pdf
- results/<tag>/network_edges.tsv
- results/<tag>/network_nodes.tsv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import corneto as cn
from corneto.methods.future.carnival import CarnivalFlow
from corneto.graph import Graph


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
FEATURE_TAGS = ["tf", "kinase", "both"]
LAMBDA_REG = 0.01
SOLVER = "SCIP"
TIME_LIMIT = 300

SIGN_ONLY = False  # keep magnitudes by default


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
try:
    _script_root = Path(__file__).resolve().parent.parent
except NameError:
    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results_new" if (_script_root / "data").is_dir() else Path("results_new")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def to_sign(x: float, eps: float = 1e-12) -> float:
    x = float(x)
    if abs(x) < eps:
        return 0.0
    return 1.0 if x > 0 else -1.0


def build_sample_data(activities_early: pd.DataFrame, secretome_early: pd.DataFrame) -> dict:
    sample_data: dict[str, dict] = {}
    sample_data["TGFB1"] = {"value": 1.0, "mapping": "vertex", "role": "input"}

    for _, row in activities_early.iterrows():
        raw = float(row["score"])
        v = to_sign(raw) if SIGN_ONLY else raw
        if SIGN_ONLY and v == 0:
            continue
        sample_data[str(row["source"])] = {"value": v, "mapping": "vertex", "role": "input"}

    for _, row in secretome_early.iterrows():
        raw = float(row["score"])
        v = to_sign(raw) if SIGN_ONLY else raw
        if SIGN_ONLY and v == 0:
            continue
        sample_data[str(row["id"])] = {"value": v, "mapping": "vertex", "role": "output"}

    return sample_data


def run_tag(tag: str) -> None:
    pkn_path = DATA_DIR / "network" / f"pkn_{tag}.tsv"
    aearly_path = DATA_DIR / "differential" / f"activities_early_{tag}.tsv"
    searly_path = DATA_DIR / "differential" / f"secretome_early_{tag}.tsv"

    pkn = pd.read_csv(pkn_path, sep="\t")
    activities_early = pd.read_csv(aearly_path, sep="\t")
    secretome_early = pd.read_csv(searly_path, sep="\t")

    print("\n" + "=" * 80)
    print(f"Running tag={tag} | PKN={pkn.shape} | inputs={activities_early.shape} | outputs={secretome_early.shape}")

    edge_tuples = list(zip(pkn["source"], pkn["mor"], pkn["target"]))
    G = Graph.from_tuples(edge_tuples)

    sample_data = build_sample_data(activities_early, secretome_early)
    data = cn.Data.from_cdict({"early": sample_data})

    n_inputs = sum(1 for v in sample_data.values() if v["role"] == "input")
    n_outputs = sum(1 for v in sample_data.values() if v["role"] == "output")
    print(f"Evidence: {n_inputs} inputs, {n_outputs} outputs (SIGN_ONLY={SIGN_ONLY})")

    carnival = CarnivalFlow(lambda_reg=LAMBDA_REG)
    problem = carnival.build(G, data)

    print(f"Solving with {SOLVER} (lambda={LAMBDA_REG}, limit={TIME_LIMIT}s)...")
    problem.solve(solver=SOLVER, verbosity=1, **{"limits/time": TIME_LIMIT})

    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    active_edges = list(np.flatnonzero(problem.expr.edge_has_signal.value))
    g = carnival.processed_graph.plot(
        custom_edge_attr=cn.pl.edge_style(problem, edge_var="edge_value"),
        custom_vertex_attr=cn.pl.vertex_style(problem, carnival.processed_graph, vertex_var="vertex_value"),
        edge_indexes=active_edges,
    )
    g.render(out_dir / "network_corneto", format="pdf", cleanup=True)

    edge_values = problem.expr.edge_value.value.flatten()
    vertex_values = problem.expr.vertex_value.value.flatten()

    edges_result = []
    for i, (src, sign, tgt) in enumerate(edge_tuples):
        if abs(edge_values[i]) > 1e-6:
            edges_result.append(
                {"source": str(src), "sign": int(sign), "target": str(tgt), "edge_value": float(edge_values[i])}
            )
    edges_df = pd.DataFrame(edges_result).sort_values(["source", "target"]).reset_index(drop=True)

    node_names = list(carnival.processed_graph.V)
    nodes_result = []
    for i, node in enumerate(node_names):
        val = float(vertex_values[i])
        if abs(val) > 1e-6:
            nodes_result.append({"node": str(node), "vertex_value": val})
    nodes_df = pd.DataFrame(nodes_result).sort_values("node").reset_index(drop=True)

    edges_df.to_csv(out_dir / "network_edges.tsv", sep="\t", index=False)
    nodes_df.to_csv(out_dir / "network_nodes.tsv", sep="\t", index=False)

    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    for tag in FEATURE_TAGS:
        run_tag(tag)