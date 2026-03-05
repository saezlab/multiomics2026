"""
Prepare CORNETO/CARNIVAL inputs for:
- TF-only
- Kinase-only
- TF+Kinase (Both)

Outputs per tag (tf / kinase / both):
- data/network/pkn_<tag>.tsv
- data/differential/activities_early_<tag>.tsv
- data/differential/secretome_early_<tag>.tsv
- data/differential/activities_late_<tag>.tsv

Also writes:
- results/inputs/feature_set_summary.tsv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
import seaborn as sns

import omnipath as op
import networkx as nx


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
FEATURE_SETS = ["TF", "Kinase", "Both"]  # exact: TF-only, Kinase-only, TF+Kinase

EARLY_TIMES = ["0.08h", "1h", "12h"]
SECRETOME_EARLY = ["0.08h", "1h", "12h", "24h"]
LATE_TIMES = ["24h", "48h", "72h", "96h"]

SIGNIFICANCE_THRESHOLD = 0.05  # activities p_value cutoff
SCORE_THRESHOLD = 3            # activities abs(score) cutoff (TF-only might need 2–2.5)
FC_THRESHOLD = float(np.log2(1.5))  # secretome abs(logFC) cutoff

N_STEPS = 5  # reachability pruning radius

REMOVE_OUTPUT_INPUT_OVERLAP = True  # avoid node being both input+output

SAVE_HEATMAPS = True
SHOW_PLOTS = False  # keep False unless running locally


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
try:
    _script_root = Path(__file__).resolve().parent.parent
except NameError:
    _script_root = Path.cwd()

DATA_DIR = _script_root / "data" if (_script_root / "data").is_dir() else Path("data")
RESULTS_DIR = _script_root / "results_new" if (_script_root / "data").is_dir() else Path("results_new")

(DATA_DIR / "differential").mkdir(exist_ok=True, parents=True)
(DATA_DIR / "network").mkdir(exist_ok=True, parents=True)
(RESULTS_DIR / "inputs").mkdir(exist_ok=True, parents=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def tagify(feature_set: str) -> str:
    fs = feature_set.strip().lower()
    if fs == "tf":
        return "tf"
    if fs == "kinase":
        return "kinase"
    if fs == "both":
        return "both"
    raise ValueError(f"Unknown feature_set: {feature_set}")


def reachable_neighbors(pkn_df: pd.DataFrame, n_steps: int, nodes: set[str], direction: str) -> pd.DataFrame:
    """Keep only edges within n_steps of nodes (downstream) or predecessors (upstream)."""
    if pkn_df.empty or not nodes:
        return pkn_df

    G = nx.DiGraph()
    G.add_edges_from(zip(pkn_df["source"], pkn_df["target"]))
    search_graph = G if direction == "downstream" else G.reverse()

    reachable = set(nodes)
    for node in nodes:
        if node in search_graph:
            lengths = nx.single_source_shortest_path_length(search_graph, node, cutoff=n_steps)
            reachable.update(lengths.keys())

    return pkn_df[pkn_df["source"].isin(reachable) & pkn_df["target"].isin(reachable)]


def save_heatmap(matrix: pd.DataFrame, title: str, cbar_label: str, outbase: Path) -> None:
    if matrix.empty:
        print(f"[heatmap] Skipping empty matrix: {title}")
        return

    vmax = float(matrix.abs().max().max())
    fig, ax = plt.subplots(figsize=(6, max(6, len(matrix) * 0.22)))
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# 1) Load data
# -----------------------------------------------------------------------------
diff_expr = pd.read_csv(DATA_DIR / "differential" / "diff_expr_all.tsv.gz", sep="\t")
activities_all = pd.read_csv(DATA_DIR / "differential" / "activities.tsv", sep="\t")

if "enzyme_type" not in activities_all.columns:
    raise ValueError("activities.tsv is missing required column: enzyme_type")

print(f"Loaded diff_expr: {diff_expr.shape}")
print(f"Loaded activities: {activities_all.shape}")
print("enzyme_type unique:", activities_all["enzyme_type"].unique())

all_genes = diff_expr["feature_id"].unique()
print(f"All measured features (feature_id): {len(all_genes)}")


# -----------------------------------------------------------------------------
# 2) Secretome candidates (base pool)
# -----------------------------------------------------------------------------
secretome_candidates = (
    diff_expr
    .query(
        "modality == 'secretomics' and time in @SECRETOME_EARLY "
        "and `adj.P.Val` < 0.05 and abs(logFC) > @FC_THRESHOLD"
    )
    .assign(abs_fc=lambda df: df["logFC"].abs())
    .sort_values("abs_fc", ascending=False)
    .drop_duplicates(subset="feature_id", keep="first")
    .rename(columns={"feature_id": "id", "logFC": "score"})
    [["id", "score"]]
    .reset_index(drop=True)
)

# Exclude TGFB1 as an output (it’s forced as an input stimulus later)
secretome_candidates = secretome_candidates[secretome_candidates["id"] != "TGFB1"].copy()
print(f"Secretome candidate pool: {len(secretome_candidates)}")


# -----------------------------------------------------------------------------
# 3) Retrieve + process OmniPath PKN ONCE
# -----------------------------------------------------------------------------
print("\nRetrieving signed interactions from OmniPath...")
pkn_raw = op.interactions.AllInteractions.get(genesymbols=True)
print(f"Total OmniPath interactions retrieved: {len(pkn_raw)}")

pkn = (
    pkn_raw[["source_genesymbol", "target_genesymbol", "consensus_stimulation", "consensus_inhibition"]]
    .copy()
)

# Keep only signed edges
pkn = pkn[pkn["consensus_stimulation"] | pkn["consensus_inhibition"]].copy()
pkn["mor"] = pkn["consensus_stimulation"].astype(int) - pkn["consensus_inhibition"].astype(int)
pkn = pkn[pkn["mor"] != 0].copy()

pkn = (
    pkn.rename(columns={"source_genesymbol": "source", "target_genesymbol": "target"})
       [["source", "mor", "target"]]
       .drop_duplicates()
)

# Keep only nodes present in our measured universe
pkn_filtered = pkn[pkn["source"].isin(all_genes) & pkn["target"].isin(all_genes)].copy()

# Add canonical TGFB edges (optional, but matches your existing approach)
tgfb_edges = pd.DataFrame(
    {
        "source": ["TGFB1"] * 9,
        "mor": [1] * 9,
        "target": ["AKT1", "PI3K", "MAPK1", "SMAD1", "SMAD2", "SMAD3", "SMAD4", "SMAD5", "MAPK14"],
    }
)
pkn_filtered = pd.concat([pkn_filtered, tgfb_edges], ignore_index=True).drop_duplicates()

print(f"Signed PKN after measured-feature filter (+ TGFB edges): {len(pkn_filtered)} edges")


# -----------------------------------------------------------------------------
# 4) Per-feature-set run
# -----------------------------------------------------------------------------
ALL_TIMES = ["0.08h", "1h", "12h", "24h", "48h", "72h", "96h"]
SECRETOME_TIMES = ["12h", "24h", "48h", "72h", "96h"]

summary_rows = []

for feature_set in FEATURE_SETS:
    tag = tagify(feature_set)
    out_inputs_dir = RESULTS_DIR / "inputs" / tag
    out_inputs_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Preparing inputs for FEATURE_SET={feature_set} (tag={tag})")

    # Filter activities by enzyme_type
    if feature_set == "Both":
        activities = activities_all.copy()  # TF + Kinase (union)
    else:
        activities = activities_all.loc[activities_all["enzyme_type"] == feature_set].copy()

    # Early activities (inputs)
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

    # Late activities (optional model variant, kept for completeness)
    activities_late = (
        activities
        .query(
            "time in @LATE_TIMES and "
            "p_value < @SIGNIFICANCE_THRESHOLD and "
            "abs(score) > @SCORE_THRESHOLD"
        )
        .assign(abs_score=lambda df: df["score"].abs())
        .sort_values("abs_score", ascending=False)
        .drop_duplicates(subset="source", keep="first")
        [["source", "score"]]
        .reset_index(drop=True)
    )

    # Secretome outputs (measurements)
    secretome_early = secretome_candidates.copy()
    if REMOVE_OUTPUT_INPUT_OVERLAP and not activities_early.empty:
        secretome_early = secretome_early[~secretome_early["id"].isin(activities_early["source"])].copy()

    if not activities_late.empty and not secretome_early.empty:
        activities_late = activities_late[~activities_late["source"].isin(secretome_early["id"])].copy()

    # Filter by PKN node coverage BEFORE pruning
    pkn_nodes = set(pkn_filtered["source"]) | set(pkn_filtered["target"])
    activities_early = activities_early[activities_early["source"].isin(pkn_nodes)].copy()
    secretome_early = secretome_early[secretome_early["id"].isin(pkn_nodes)].copy()
    activities_late = activities_late[activities_late["source"].isin(pkn_nodes)].copy()

    print(f"inputs activities_early: {len(activities_early)}")
    print(f"outputs secretome_early: {len(secretome_early)}")
    print(f"late activities_late:    {len(activities_late)}")

    # Heatmaps (optional)
    if SAVE_HEATMAPS:
        all_selected = set(activities_early["source"]) | set(activities_late["source"])
        if all_selected:
            act_matrix = (
                activities
                .query("source in @all_selected")
                .pivot(index="source", columns="time", values="score")
                .reindex(columns=ALL_TIMES)
            )
            act_matrix = act_matrix.loc[act_matrix.abs().max(axis=1).sort_values(ascending=False).index]
            save_heatmap(
                act_matrix,
                title=f"Selected activities ({feature_set})",
                cbar_label="Activity score",
                outbase=out_inputs_dir / "activities_heatmap",
            )

        if not secretome_early.empty:
            sec_matrix = (
                diff_expr
                .query("modality == 'secretomics' and feature_id in @secretome_early['id'].values")
                .pivot(index="feature_id", columns="time", values="logFC")
                .reindex(columns=SECRETOME_TIMES)
            )
            sec_matrix = sec_matrix.loc[sec_matrix.abs().max(axis=1).sort_values(ascending=False).index]
            save_heatmap(
                sec_matrix,
                title=f"Secretome fold changes (outputs) [{feature_set}]",
                cbar_label="log₂ FC",
                outbase=out_inputs_dir / "secretome_heatmap",
            )

    # PKN pruning (reachability depends on input/output sets)
    all_input_nodes = set(activities_early["source"]) | {"TGFB1"}
    all_output_nodes = set(secretome_early["id"])

    pkn_pruned = pkn_filtered.copy()
    prev_size = -1
    while len(pkn_pruned) != prev_size:
        prev_size = len(pkn_pruned)
        pkn_pruned = reachable_neighbors(pkn_pruned, N_STEPS, all_input_nodes, "downstream")
        pkn_pruned = reachable_neighbors(pkn_pruned, N_STEPS, all_output_nodes, "upstream")

        pkn_nodes_pruned = set(pkn_pruned["source"]) | set(pkn_pruned["target"])
        all_input_nodes &= pkn_nodes_pruned
        all_output_nodes &= pkn_nodes_pruned

    pkn_nodes_pruned = set(pkn_pruned["source"]) | set(pkn_pruned["target"])

    # Final filter after pruning
    activities_early = activities_early[activities_early["source"].isin(pkn_nodes_pruned)].copy()
    secretome_early = secretome_early[secretome_early["id"].isin(pkn_nodes_pruned)].copy()
    activities_late = activities_late[activities_late["source"].isin(pkn_nodes_pruned)].copy()

    # Save
    pkn_path = DATA_DIR / "network" / f"pkn_{tag}.tsv"
    aearly_path = DATA_DIR / "differential" / f"activities_early_{tag}.tsv"
    searly_path = DATA_DIR / "differential" / f"secretome_early_{tag}.tsv"
    alate_path = DATA_DIR / "differential" / f"activities_late_{tag}.tsv"

    pkn_pruned.to_csv(pkn_path, sep="\t", index=False)
    activities_early.to_csv(aearly_path, sep="\t", index=False)
    secretome_early.to_csv(searly_path, sep="\t", index=False)
    activities_late.to_csv(alate_path, sep="\t", index=False)

    print(f"Saved tag={tag}: PKN edges={len(pkn_pruned)}, nodes={len(pkn_nodes_pruned)}")

    summary_rows.append(
        {
            "feature_set": feature_set,
            "tag": tag,
            "n_inputs_activities_early": int(len(activities_early)),
            "n_outputs_secretome_early": int(len(secretome_early)),
            "n_late_activities": int(len(activities_late)),
            "pkn_edges": int(len(pkn_pruned)),
            "pkn_nodes": int(len(pkn_nodes_pruned)),
        }
    )

summary_df = pd.DataFrame(summary_rows)
summary_path = RESULTS_DIR / "inputs" / "feature_set_summary.tsv"
summary_df.to_csv(summary_path, sep="\t", index=False)

print("\nWrote summary:", summary_path)
print(summary_df.to_string(index=False))