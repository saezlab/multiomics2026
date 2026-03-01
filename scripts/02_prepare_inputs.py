"""Data preparation and prior knowledge network retrieval.

This script prepares all inputs needed for CORNETO network inference:
1. Load differential expression and enzyme activity data
2. Select upstream (perturbation) and downstream (measurement) nodes
3. Retrieve the prior knowledge network (PKN) from OmniPath
4. Filter and prune the PKN for the relevant subnetwork
5. Save all prepared inputs
"""

import pandas as pd
import numpy as np
import omnipath as op
from pathlib import Path

# %% Setting up paths for data and results
# this is not important, it's we only needed to allow us to run the same code
# either as a script or in an interactive session

try:
    _script_root = Path(__file__).resolve().parent.parent
except NameError:
    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results" if (_script_root / "data").is_dir() else Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# %% 1. Load data

diff_expr = pd.read_csv(DATA_DIR / "differential" / "diff_expr_all.tsv", sep="\t")
activities = pd.read_csv(DATA_DIR / "differential" / "activities.tsv", sep="\t")

print(f"Differential expression: {len(diff_expr)} rows")
print(f"Activities: {len(activities)} rows")

# %% 2. All gene symbols detected across omics

all_genes = diff_expr["feature_id"].unique()
print(f"\nAll measured genes/proteins: {len(all_genes)}")

# %% 3. Select perturbation nodes: TF and kinase activities
#
# In the paper's "early" model (model 2), enzyme activities together with
# TGFB1 serve as perturbations (inputs) to the network. The idea is that
# we treat the inferred TF/kinase activity states as known, and ask CARNIVAL
# to find a subnetwork explaining the secretome changes.
#
# We select significant enzyme activities from early time points
# (5 min, 1 h, 12 h). For each enzyme, we take the time point with
# the highest |score|.

EARLY_TIMES = ["0.08h", "1h", "12h"]
SIGNIFICANCE_THRESHOLD = 0.03
SCORE_THRESHOLD = 3

activities_early = (
    activities
    .query(
        "time in @EARLY_TIMES and "
        "p_value < @SIGNIFICANCE_THRESHOLD and "
        "abs(score) > @SCORE_THRESHOLD"
    )
    .assign(abs_score=lambda df: df["score"].abs())
    .sort_values("abs_score", ascending=False)
    .drop_duplicates(subset="source", keep="first")
    [["source", "score"]]
    .reset_index(drop=True)
)

print(f"\nPerturbation enzymes (early, significant): {len(activities_early)}")
print(activities_early.head(10))

# %% 4. Select measurement nodes: secretome fold changes
#
# Secreted proteins serve as the measurements (outputs) that the network
# should explain. These are the observed fold changes in the secretome
# upon TGF-beta stimulation.

SECRETOME_EARLY = ["0.08h", "1h", "12h", "24h"]
FC_THRESHOLD = np.log2(1.5)

secretome_early = (
    diff_expr
    .query("modality == 'secretomics' and time in @SECRETOME_EARLY "
           "and `adj.P.Val` < 0.05 and abs(logFC) > @FC_THRESHOLD")
    .assign(abs_fc=lambda df: df["logFC"].abs())
    .sort_values("abs_fc", ascending=False)
    .drop_duplicates(subset="feature_id", keep="first")
    .rename(columns={"feature_id": "id", "logFC": "score"})
    [["id", "score"]]
    .reset_index(drop=True)
)

# Remove TGFB1 if present (it's a stimulus ligand, not a measurement here)
secretome_early = secretome_early[secretome_early["id"] != "TGFB1"]
# Remove overlap with perturbation enzymes
secretome_early = secretome_early[~secretome_early["id"].isin(activities_early["source"])]

print(f"\nSecretome measurements: {len(secretome_early)}")
print(secretome_early.head(10))

# %% 5. Retrieve prior knowledge network from OmniPath
#
# We get all signed, directed protein-protein interactions.

print("\nRetrieving interactions from OmniPath...")
pkn_raw = op.interactions.AllInteractions.get(genesymbols=True)

print(f"Total initial PKN from OmniPath: {len(pkn_raw)}")

# %% 6. Process PKN: select signed interactions with gene symbols
#
# With genesymbols=True, the dataframe has `source_genesymbol` and
# `target_genesymbol` columns alongside the UniProt ID columns.

pkn = (
    pkn_raw
    [["source_genesymbol", "target_genesymbol",
      "consensus_stimulation", "consensus_inhibition"]]
    .copy()
)

# Remove interactions without clear consensus sign
pkn = pkn[pkn["consensus_stimulation"] | pkn["consensus_inhibition"]]

# Compute mode of regulation: +1 (activation) or -1 (inhibition)
pkn["mor"] = pkn["consensus_stimulation"].astype(int) - pkn["consensus_inhibition"].astype(int)
pkn = pkn[pkn["mor"] != 0]

# Create final column layout or CORNETO
pkn = (
    pkn
    .rename(columns={
        "source_genesymbol": "source",
        "target_genesymbol": "target",
    })
    [["source", "mor", "target"]]
    .drop_duplicates()
)

print(f"\nSigned PKN: {len(pkn)} interactions")
print(f"  Nodes: {len(set(pkn['source']) | set(pkn['target']))}")

# %% 7. Filter PKN for expressed genes
#
# Keep only interactions where both source and target were detected
# in at least one omics modality.

pkn_filtered = pkn[
    pkn["source"].isin(all_genes) & pkn["target"].isin(all_genes)
]

print(f"\nPKN filtered for expressed genes: {len(pkn_filtered)} interactions")
print(f"  Nodes: {len(set(pkn_filtered['source']) | set(pkn_filtered['target']))}")

# %% 8. Add known TGF-beta signaling edges
#
# Ensure canonical TGFB1 -> SMAD pathway edges are represented in the PKN.
# These may help the network connect perturbation and measurement nodes.
# We do this following the "Network modeling" section in the Methods of the
# paper.
# Actually it adds only one interaction (TGFB1 -> PI3K), the rest already
# present in OmniPath.

tgfb_edges = pd.DataFrame({
    "source": ["TGFB1"] * 9,
    "mor": [1] * 9,
    "target": ["AKT1", "PI3K", "MAPK1", "SMAD1", "SMAD2",
               "SMAD3", "SMAD4", "SMAD5", "MAPK14"],
})

pkn_filtered = pd.concat([pkn_filtered, tgfb_edges]).drop_duplicates()
print(f"\nPKN with TGF-beta edges: {len(pkn_filtered)} interactions")

# %% 9. Filter inputs for PKN coverage
#
# Keep only perturbation/measurement nodes that appear in the PKN.
# This also demonstrates how prior-knowledge availability limits discovery:
# some of our hits are not present in the PKN, we don't have causal network
# knowledge about them, so we simply can not create mechanistic hypotheses
# about them.

pkn_nodes = set(pkn_filtered["source"]) | set(pkn_filtered["target"])

activities_early = activities_early[activities_early["source"].isin(pkn_nodes)]
secretome_early = secretome_early[secretome_early["id"].isin(pkn_nodes)]

print(f"\nAfter PKN filtering:")
print(f"  Perturbation enzymes: {len(activities_early)}")
print(f"  Secretome measurements: {len(secretome_early)}")

# %% 10. Prune PKN for reachability
#
# We iteratively remove nodes that cannot be reached from perturbation
# nodes (controllability) or that have no path to measurement nodes
# (observability). This focuses the network on relevant biology.

import networkx as nx

N_STEPS = 5


def reachable_neighbors(pkn_df, n_steps, nodes, direction="downstream"):
    """Keep only PKN edges within n_steps of the given nodes.

    direction: "downstream" follows edges forward from the nodes,
               "upstream" follows edges backward to find predecessors.
    """
    G = nx.DiGraph()
    G.add_edges_from(zip(pkn_df["source"], pkn_df["target"]))
    search_graph = G if direction == "downstream" else G.reverse()
    reachable = set(nodes)
    for node in nodes:
        if node in search_graph:
            lengths = nx.single_source_shortest_path_length(
                search_graph, node, cutoff=n_steps,
            )
            reachable.update(lengths.keys())
    return pkn_df[pkn_df["source"].isin(reachable) & pkn_df["target"].isin(reachable)]


all_input_nodes = set(activities_early["source"]) | {"TGFB1"}
all_output_nodes = set(secretome_early["id"])

# Iteratively prune until stable
pkn_pruned = pkn_filtered.copy()
prev_size = 0

while len(pkn_pruned) != prev_size:
    prev_size = len(pkn_pruned)
    pkn_pruned = reachable_neighbors(pkn_pruned, N_STEPS, all_input_nodes, "downstream")
    pkn_pruned = reachable_neighbors(pkn_pruned, N_STEPS, all_output_nodes, "upstream")
    # Update input/output nodes to those still in the network
    pkn_nodes = set(pkn_pruned["source"]) | set(pkn_pruned["target"])
    all_input_nodes = all_input_nodes & pkn_nodes
    all_output_nodes = all_output_nodes & pkn_nodes

# Final filter of enzymes and secretome
activities_early = activities_early[activities_early["source"].isin(pkn_nodes)]
secretome_early = secretome_early[secretome_early["id"].isin(pkn_nodes)]

print(f"\nPruned PKN (reachability within {N_STEPS} steps):")
print(f"  Interactions: {len(pkn_pruned)}")
print(f"  Nodes: {len(pkn_nodes)}")
print(f"  Perturbation enzymes: {len(activities_early)}")
print(f"  Secretome measurements: {len(secretome_early)}")

# %% 11. Save prepared inputs

pkn_pruned.to_csv(DATA_DIR / "network" / "pkn.tsv", sep="\t", index=False)
activities_early.to_csv(DATA_DIR / "differential" / "activities_early.tsv", sep="\t", index=False)
secretome_early.to_csv(DATA_DIR / "differential" / "secretome_early.tsv", sep="\t", index=False)

print(f"\nSaved prepared inputs:")
print(f"  {DATA_DIR / 'network' / 'pkn.tsv'}")
print(f"  {DATA_DIR / 'differential' / 'activities_early.tsv'}")
print(f"  {DATA_DIR / 'differential' / 'secretome_early.tsv'}")

# %% Summary
#
# We now have three key inputs for CORNETO:
#
# 1. pkn.tsv: The pruned prior knowledge network (source, mor, target)
# 2. activities_early.tsv: Perturbation scores (TF + kinase activities)
# 3. secretome_early.tsv: Measurement scores (secretome fold changes)
#
# In addition, TGFB1 = +1 is the known stimulus.
#
# The next step (script 03) will use these to run CARNIVAL network inference.
