"""
CARNIVAL network inference with CORNETO.

This script uses the CARNIVAL algorithm (implemented in CORNETO) to find
an optimal subnetwork of the prior knowledge network that connects
perturbations to measurements.

In our "early" model (corresponding to model 2 in the paper):
- Perturbations (inputs): TGFB1 stimulus + early TF/kinase activities
- Measurements (outputs): early secretome fold changes

CARNIVAL solves an integer linear program that:
- Maximizes consistency between network edges and observed data
- Minimizes network size (L0 penalty on active edges)
"""

import pandas as pd
import numpy as np
from pathlib import Path

import corneto as cn
from corneto.methods.future.carnival import CarnivalFlow
from corneto.graph import Graph

# %% Setting up paths for data and results
# this is not important, it's we only needed to allow us to run the same code
# either as a script or in an interactive session

#try:
_script_root = Path(__file__).resolve().parent.parent
#except NameError:
#    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results" if (_script_root / "data").is_dir() else Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# %% 1. Load prepared inputs

pkn = pd.read_csv(DATA_DIR / "network" / "pkn.tsv", sep="\t")
activities_early = pd.read_csv(DATA_DIR / "differential" / "activities_early.tsv", sep="\t")
secretome_early = pd.read_csv(DATA_DIR / "differential" / "secretome_early.tsv", sep="\t")

print(f"PKN: {len(pkn)} interactions, "
      f"{len(set(pkn['source']) | set(pkn['target']))} nodes")
print(f"Perturbation nodes: {len(activities_early)}")
print(f"Measurements (secretome): {len(secretome_early)}")

# %% 2. Build CORNETO Graph from PKN
#
# The Graph object represents the prior knowledge network.
# Each edge is a tuple: (source, sign, target)

edge_tuples = list(zip(pkn["source"], pkn["mor"], pkn["target"]))
G = cn.graph.Graph.from_tuples(edge_tuples)

print(f"\nCORNETO Graph: {G.num_vertices} vertices, {G.num_edges} edges")

# %% 3. Prepare perturbation and measurement data
#
# CORNETO expects a dictionary of samples, where each sample maps
# node names to their role (input/output) and observed value.
#
# - Inputs (perturbations): TGFB1 stimulus and TF/kinase activity scores
# - Outputs (measurements): secretome fold changes to be explained

sample_data = {}

# Add TGFB1 as the known stimulus
sample_data["TGFB1"] = {"value": 1.0, "mapping": "vertex", "role": "input"}

# Add enzyme activities as perturbations
for _, row in activities_early.iterrows():
    sample_data[row["source"]] = {
        "value": float(row["score"]),
        "mapping": "vertex",
        "role": "input",
    }

# Add secretome as measurements
for _, row in secretome_early.iterrows():
    sample_data[row["id"]] = {
        "value": float(row["score"]),
        "mapping": "vertex",
        "role": "output",
    }

data = cn.Data.from_cdict({"early": sample_data})

n_inputs = sum(1 for v in sample_data.values() if v["role"] == "input")
n_outputs = sum(1 for v in sample_data.values() if v["role"] == "output")
print(f"\nData: {n_inputs} inputs, {n_outputs} outputs")

# %% 4. Build CARNIVAL optimization problem
#
# CARNIVAL finds the smallest subnetwork of the PKN that is logically
# consistent with the observed perturbation and measurement data.
# The optimization has two objectives:
#
#   Objective 0 (data fit): how well do the network edge signs agree
#       with the observed activity scores? Each measurement node has
#       an observed sign (+/-); the network should propagate signals
#       from perturbations to measurements without contradictions.
#       Lower = better fit (fewer sign mismatches).
#
#   Objective 1 (network size): how many edges are active? This is the
#       L0 regularization term — "L0" simply means we count the number
#       of non-zero (active) edges. We want a parsimonious network:
#       the simplest explanation consistent with the data.
#
# lambda_reg controls the trade-off between these two objectives:
# - Higher lambda_reg -> sparser network (fewer edges, possibly worse fit)
# - Lower lambda_reg -> denser network (better fit, harder to interpret)
#
# CarnivalFlow is a specific CARNIVAL formulation that models signal
# propagation as network flow, which makes the optimization more
# efficient for large networks.

def run_optimization(lamBda, solver, time_limit, g , data):
    print(f"\nNetwork optimization: lambda={lamBda}, solver={solver}, time_limit={time_limit}")
    carnival = CarnivalFlow(lambda_reg=lamBda)
    problem = carnival.build(g, data)

    print(f"\nCARNIVAL problem built (lambda_reg={lamBda})")

    print(f"\nSolving with {solver} (time limit: {time_limit}s)...")
    problem.solve(solver=solver, verbosity=1, **{"limits/time": time_limit})

    # Objective 0: data fit (sign mismatches); Objective 1: network size (edge count)
    print("\nObjective values:")
    for i, obj in enumerate(problem.objectives):
        print(f"  Objective {i}: {obj.value}")

    active_edges = list(np.flatnonzero(problem.expr.edge_has_signal.value))

    gr = carnival.processed_graph.plot(
        custom_edge_attr=cn.pl.edge_style(problem, edge_var="edge_value"),
        custom_vertex_attr=cn.pl.vertex_style(problem, carnival.processed_graph,
                                            vertex_var="vertex_value"),
        edge_indexes=active_edges)  # install graphviz first


    gr.render(RESULTS_DIR / f"network_corneto_{lamBda}_{solver}_{time_limit}", format="pdf", cleanup=True)
    print(f"Saved network plot to {RESULTS_DIR / 'network_corneto_{lamBda}_{solver}_{time_limit}.pdf'}")

    edge_values = problem.expr.edge_value.value.flatten()
    vertex_values = problem.expr.vertex_value.value.flatten()

    edges_result = []

    for i, (src, sign, tgt) in enumerate(edge_tuples):
        if abs(edge_values[i]) > 1e-6:  # active edges only
            edges_result.append({
                "source": src,
                "sign": int(sign),
                "target": tgt,
                "edge_value": float(edge_values[i]),
            })


    edges_df = pd.DataFrame(edges_result).sort_values(["source", "target"]).reset_index(drop=True)
    print(f"\nActive edges: {len(edges_df)}")

    # Build node result table
    vertex_names = carnival.processed_graph.V
    nodes_result = []

    for i, name in enumerate(vertex_names):
        if abs(vertex_values[i]) > 1e-6:  # active nodes only
            node_type = "intermediate"
            if name in sample_data:
                node_type = sample_data[name]["role"]
            nodes_result.append({
                "node": name,
                "value": float(vertex_values[i]),
                "type": node_type,
            })

    nodes_df = pd.DataFrame(nodes_result).sort_values("node").reset_index(drop=True)
    print(f"Active nodes: {len(nodes_df)}")

    edges_df.to_csv(RESULTS_DIR / f"network_edges_{lamBda}_{solver}_{time_limit}.tsv", sep="\t", index=False)
    nodes_df.to_csv(RESULTS_DIR / f"network_nodes_{lamBda}_{solver}_{time_limit}.tsv", sep="\t", index=False)

    print(f"\nSaved results to {RESULTS_DIR}:")
    print(f"  network_edges_{lamBda}_{solver}_{time_limit}.tsv ({len(edges_df)} edges)")
    print(f"  network_nodes_{lamBda}_{solver}_{time_limit}.tsv ({len(nodes_df)} nodes)")

    if len(edges_df) > 0:
        print(f"\nNetwork summary: lambda={lamBda}, solver={solver}, time_limit={time_limit}")
        print(f"  Activating edges: {(edges_df['sign'] > 0).sum()}")
        print(f"  Inhibiting edges: {(edges_df['sign'] < 0).sum()}")
        print(f"  Input nodes: {(nodes_df['type'] == 'input').sum()}")
        print(f"  Output nodes: {(nodes_df['type'] == 'output').sum()}")
        print(f"  Intermediate nodes: {(nodes_df['type'] == 'intermediate').sum()}")
    else:
        print("\nNo active edges found. Consider adjusting lambda_reg or solver settings.")
    return None

lambda_values = [0.01,0.001]#[0.1,0.01,0.001]
solver = ["CVXOPT", "GLPK", "GLPK_MI", "SCIP"] #["CVXOPT", "GLPK", "GLPK_MI", "HIGHS", "SCIP", "SCIPY"]
time_limit = [300,500]

for lval in lambda_values:
    for solv in solver:
        for timel in time_limit:
            run_optimization(lval,solv,timel, G, data)

# %% Notes
#
# If the network is too sparse, try:
#   - Decreasing lambda_reg (e.g., 0.001)
#   - Increasing the time limit
#   - Using a different solver (e.g., "scip" for better MIP performance)
#
# If the network is too dense, try:
#   - Increasing lambda_reg (e.g., 0.1)
#   - Applying stricter significance thresholds for inputs/outputs
#
# The next script (04) will visualize the resulting network.
