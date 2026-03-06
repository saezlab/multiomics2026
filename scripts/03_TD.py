"""
CARNIVAL network inference with CORNETO (TF-only / Kinase-only / Both).

Runs two models following the paper, per feature-set:
- Model 1 (TGFB1 → early activities)
- Model 2 (activities + TGFB1 → early secretome)

Then merges model1+model2 into a combined early network.

Writes per tag to:
results/<tag>/
  model1_edges.tsv, model1_nodes.tsv, network_model1.pdf
  model2_edges.tsv, model2_nodes.tsv, network_model2.pdf
  network_edges.tsv, network_nodes.tsv, network_merged.pdf
"""

import pandas as pd
from pathlib import Path

from carnival_utils import (
    run_carnival, extract_results, plot_network,
    save_results, merge_networks, print_summary,
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
try:
    _script_root = Path(__file__).resolve().parent.parent
except NameError:
    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results_new" if (_script_root / "data").is_dir() else Path("results_new")
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
TAGS = ["tf", "kinase", "both"]

LAMBDA_REG = 0.001
SOLVER = "HIGHS"
TIME_LIMIT = 300  # seconds

# -----------------------------------------------------------------------------
# Run per tag
# -----------------------------------------------------------------------------
for tag in TAGS:
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"RUNNING FEATURE TAG: {tag}")
    print("=" * 80)

    # Load prepared inputs
    activities_early = pd.read_csv(DATA_DIR / "differential" / f"activities_early_{tag}.tsv", sep="\t")
    secretome_early = pd.read_csv(DATA_DIR / "differential" / f"secretome_early_{tag}.tsv", sep="\t")

    print(f"Perturbation nodes (early activities): {len(activities_early)}")
    print(f"Measurements (secretome): {len(secretome_early)}")

    # -------------------------------------------------------------------------
    # MODEL 1: TGFB1 → early activities
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("MODEL 1: TGFB1 → early activities")
    print("-" * 60)

    pkn_m1 = pd.read_csv(DATA_DIR / "network" / f"pkn_model1_{tag}.tsv", sep="\t")
    pkn_m1_nodes = set(pkn_m1["source"]) | set(pkn_m1["target"])

    sample_data_m1 = {
        "TGFB1": {"value": 1.0, "mapping": "vertex", "role": "input"},
    }
    for _, row in activities_early.iterrows():
        if row["source"] in pkn_m1_nodes:
            sample_data_m1[row["source"]] = {
                "value": float(row["score"]),
                "mapping": "vertex",
                "role": "output",
            }

    carnival_m1, problem_m1, data_m1, _ = run_carnival(
        pkn_m1, sample_data_m1, lambda_reg=LAMBDA_REG,
        solver=SOLVER, time_limit=TIME_LIMIT,
    )
    edges_m1, nodes_m1 = extract_results(carnival_m1, problem_m1, sample_data_m1)

    print("\nModel 1 result:")
    print_summary(edges_m1, nodes_m1)

    save_results(edges_m1, nodes_m1, "model1", out_dir)
    g_m1 = plot_network(edges_m1, nodes_m1)
    g_m1.render(out_dir / "network_model1", format="pdf", cleanup=True)

    # -------------------------------------------------------------------------
    # MODEL 2: activities + TGFB1 → early secretome
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("MODEL 2: activities + TGFB1 → early secretome")
    print("-" * 60)

    pkn_m2 = pd.read_csv(DATA_DIR / "network" / f"pkn_model2_{tag}.tsv", sep="\t")
    pkn_m2_nodes = set(pkn_m2["source"]) | set(pkn_m2["target"])

    sample_data_m2 = {
        "TGFB1": {"value": 1.0, "mapping": "vertex", "role": "input"},
    }
    for _, row in activities_early.iterrows():
        if row["source"] in pkn_m2_nodes:
            sample_data_m2[row["source"]] = {
                "value": float(row["score"]),
                "mapping": "vertex",
                "role": "input",
            }
    for _, row in secretome_early.iterrows():
        if row["id"] in pkn_m2_nodes:
            sample_data_m2[row["id"]] = {
                "value": float(row["score"]),
                "mapping": "vertex",
                "role": "output",
            }

    carnival_m2, problem_m2, data_m2, _ = run_carnival(
        pkn_m2, sample_data_m2, lambda_reg=LAMBDA_REG,
        solver=SOLVER, time_limit=TIME_LIMIT,
    )
    edges_m2, nodes_m2 = extract_results(carnival_m2, problem_m2, sample_data_m2)

    print("\nModel 2 result:")
    print_summary(edges_m2, nodes_m2)

    save_results(edges_m2, nodes_m2, "model2", out_dir)
    g_m2 = plot_network(edges_m2, nodes_m2)
    g_m2.render(out_dir / "network_model2", format="pdf", cleanup=True)

    # -------------------------------------------------------------------------
    # MERGE: combined early network = union(model1, model2)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("MERGED: combined early network (model1 ∪ model2)")
    print("-" * 60)

    edges_merged, nodes_merged = merge_networks([edges_m1, edges_m2], [nodes_m1, nodes_m2])

    print("\nMerged network:")
    print_summary(edges_merged, nodes_merged)

    save_results(edges_merged, nodes_merged, "network", out_dir)
    g_merged = plot_network(edges_merged, nodes_merged)
    g_merged.render(out_dir / "network_merged", format="pdf", cleanup=True)

    print(f"\nSaved all outputs to: {out_dir}")