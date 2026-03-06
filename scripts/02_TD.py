"""Data preparation and prior knowledge network retrieval (TF-only / Kinase-only / Both).

This script prepares all inputs needed for CORNETO network inference:
1. Load differential expression and enzyme activity data
2. Select upstream (perturbation) and downstream (measurement) nodes
3. Retrieve the prior knowledge network (PKN) from OmniPath
4. Filter and prune the PKN separately for each model (1/2/3)
5. Save all prepared inputs PER FEATURE-SET (tf/kinase/both)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import omnipath as op
from pathlib import Path
import networkx as nx

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

(DATA_DIR / "network").mkdir(exist_ok=True, parents=True)
(DATA_DIR / "differential").mkdir(exist_ok=True, parents=True)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
FEATURE_SETS = ["TF", "Kinase", "Both"]  # Both = TF + Kinase union

EARLY_TIMES = ["0.08h", "1h", "12h"]
SECRETOME_EARLY = ["0.08h", "1h", "12h", "24h"]
LATE_TIMES = ["24h", "48h", "72h", "96h"]
ALL_TIMES = ["0.08h", "1h", "12h", "24h", "48h", "72h", "96h"]
SECRETOME_TIMES = ["12h", "24h", "48h", "72h", "96h"]

SIGNIFICANCE_THRESHOLD = 0.05
SCORE_THRESHOLD = 3
FC_THRESHOLD = np.log2(1.5)

N_STEPS = 5

SHOW_PLOTS = True  # set False if running headless

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def tagify(fs: str) -> str:
    fs = fs.strip().lower()
    if fs == "tf":
        return "tf"
    if fs == "kinase":
        return "kinase"
    if fs == "both":
        return "both"
    raise ValueError(f"Unknown FEATURE_SET: {fs}")

def reachable_neighbors(pkn_df, n_steps, nodes, direction="downstream"):
    """Keep only PKN edges within n_steps of the given nodes.

    direction: "downstream" follows edges forward from the nodes,
               "upstream" follows edges backward to find predecessors.
    """
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

def prune_pkn(pkn_df, input_nodes, output_nodes, n_steps):
    """Iteratively prune PKN for reachability between inputs and outputs."""
    input_nodes = set(input_nodes)
    output_nodes = set(output_nodes)
    pkn_p = pkn_df.copy()
    prev_size = -1

    while len(pkn_p) != prev_size:
        prev_size = len(pkn_p)
        pkn_p = reachable_neighbors(pkn_p, n_steps, input_nodes, "downstream")
        pkn_p = reachable_neighbors(pkn_p, n_steps, output_nodes, "upstream")
        pkn_nodes = set(pkn_p["source"]) | set(pkn_p["target"])
        input_nodes = input_nodes & pkn_nodes
        output_nodes = output_nodes & pkn_nodes

    pkn_nodes = set(pkn_p["source"]) | set(pkn_p["target"])
    return pkn_p, pkn_nodes

def save_heatmap(df_mat, title, cbar_label, out_pdf, out_png):
    if df_mat.empty:
        print(f"[heatmap] Skipping empty heatmap: {title}")
        return
    vmax = df_mat.abs().max().max()
    fig, ax = plt.subplots(figsize=(5, max(6, len(df_mat) * 0.22)))
    sns.heatmap(
        df_mat, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.5, ax=ax, cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

# -----------------------------------------------------------------------------
# 1) Load data
# -----------------------------------------------------------------------------
diff_expr = pd.read_csv(DATA_DIR / "differential" / "diff_expr_all.tsv.gz", sep="\t")
activities_all = pd.read_csv(DATA_DIR / "differential" / "activities.tsv", sep="\t")

print(f"Differential expression: {len(diff_expr)} rows")
print(f"Activities: {len(activities_all)} rows")

if "enzyme_type" not in activities_all.columns:
    raise ValueError("activities.tsv must contain column: enzyme_type (TF/Kinase)")

all_genes = diff_expr["feature_id"].unique()
print(f"\nAll measured genes/proteins: {len(all_genes)}")

# -----------------------------------------------------------------------------
# 2) Retrieve and process OmniPath PKN ONCE
# -----------------------------------------------------------------------------
print("\nRetrieving interactions from OmniPath...")
pkn_raw = op.interactions.AllInteractions.get(genesymbols=True)
print(f"Total initial PKN from OmniPath: {len(pkn_raw)}")

pkn = (
    pkn_raw[["source_genesymbol", "target_genesymbol",
             "consensus_stimulation", "consensus_inhibition"]]
    .copy()
)

pkn = pkn[pkn["consensus_stimulation"] | pkn["consensus_inhibition"]]
pkn["mor"] = pkn["consensus_stimulation"].astype(int) - pkn["consensus_inhibition"].astype(int)
pkn = pkn[pkn["mor"] != 0]

pkn = (
    pkn.rename(columns={"source_genesymbol": "source", "target_genesymbol": "target"})
       [["source", "mor", "target"]]
       .drop_duplicates()
)

pkn_filtered = pkn[pkn["source"].isin(all_genes) & pkn["target"].isin(all_genes)].copy()

tgfb_edges = pd.DataFrame({
    "source": ["TGFB1"] * 9,
    "mor": [1] * 9,
    "target": ["AKT1", "PI3K", "MAPK1", "SMAD1", "SMAD2",
               "SMAD3", "SMAD4", "SMAD5", "MAPK14"],
})
pkn_filtered = pd.concat([pkn_filtered, tgfb_edges], ignore_index=True).drop_duplicates()

pkn_nodes_all = set(pkn_filtered["source"]) | set(pkn_filtered["target"])
print(f"\nPKN filtered (expressed + TGFB edges): {len(pkn_filtered)} interactions")
print(f"  Nodes: {len(pkn_nodes_all)}")

# -----------------------------------------------------------------------------
# 3) Run per feature-set
# -----------------------------------------------------------------------------
summary_rows = []

for feature_set in FEATURE_SETS:
    tag = tagify(feature_set)
    print("\n" + "=" * 80)
    print(f"FEATURE_SET={feature_set} (tag={tag})")

    # Filter activities by enzyme_type
    if feature_set == "Both":
        activities = activities_all.copy()
    else:
        activities = activities_all[activities_all["enzyme_type"] == feature_set].copy()

    # --- Early activities (perturbations for model 2; outputs for model 1)
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

    # --- Secretome early (measurements for model 2; inputs for model 3)
    secretome_early = (
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
    secretome_early = secretome_early[secretome_early["id"] != "TGFB1"]
    secretome_early = secretome_early[~secretome_early["id"].isin(activities_early["source"])]

    # --- Late activities (measurements for model 3)
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
    activities_late = activities_late[~activities_late["source"].isin(secretome_early["id"])]

    # --- Filter hits for PKN coverage
    activities_early = activities_early[activities_early["source"].isin(pkn_nodes_all)].copy()
    secretome_early = secretome_early[secretome_early["id"].isin(pkn_nodes_all)].copy()
    activities_late = activities_late[activities_late["source"].isin(pkn_nodes_all)].copy()

    print(f"Perturbation enzymes (early): {len(activities_early)}")
    print(f"Secretome measurements (early): {len(secretome_early)}")
    print(f"Late measurement enzymes: {len(activities_late)}")

    # --- Heatmaps (saved per tag so no overwrites)
    out_dir = RESULTS_DIR / "inputs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    all_selected = set(activities_early["source"]) | set(activities_late["source"])
    if len(all_selected) > 0:
        act_matrix = (
            activities
            .query("source in @all_selected")
            .pivot(index="source", columns="time", values="score")
            .reindex(columns=ALL_TIMES)
        )
        act_matrix = act_matrix.loc[act_matrix.abs().max(axis=1).sort_values(ascending=False).index]
        save_heatmap(
            act_matrix,
            title=f"Significant activities ({feature_set})",
            cbar_label="Activity score",
            out_pdf=out_dir / "activities_heatmap.pdf",
            out_png=out_dir / "activities_heatmap.png",
        )

    if len(secretome_early) > 0:
        sec_matrix = (
            diff_expr
            .query("modality == 'secretomics' and feature_id in @secretome_early['id'].values")
            .pivot(index="feature_id", columns="time", values="logFC")
            .reindex(columns=SECRETOME_TIMES)
        )
        sec_matrix = sec_matrix.loc[sec_matrix.abs().max(axis=1).sort_values(ascending=False).index]
        save_heatmap(
            sec_matrix,
            title=f"Secretome fold changes (measurements) ({feature_set})",
            cbar_label="log₂ FC",
            out_pdf=out_dir / "secretome_heatmap.pdf",
            out_png=out_dir / "secretome_heatmap.png",
        )

    # --- Prune PKN per model (per feature-set)
    pkn_model1, pkn_nodes_m1 = prune_pkn(
        pkn_filtered,
        input_nodes={"TGFB1"},
        output_nodes=set(activities_early["source"]),
        n_steps=N_STEPS,
    )

    pkn_model2, pkn_nodes_m2 = prune_pkn(
        pkn_filtered,
        input_nodes=set(activities_early["source"]) | {"TGFB1"},
        output_nodes=set(secretome_early["id"]),
        n_steps=N_STEPS,
    )

    pkn_model3, pkn_nodes_m3 = prune_pkn(
        pkn_filtered,
        input_nodes=set(secretome_early["id"]) | {"TGFB1"},
        output_nodes=set(activities_late["source"]),
        n_steps=N_STEPS,
    )

    # --- Save prepared inputs PER TAG
    (DATA_DIR / "network").mkdir(exist_ok=True, parents=True)
    (DATA_DIR / "differential").mkdir(exist_ok=True, parents=True)

    pkn_model1.to_csv(DATA_DIR / "network" / f"pkn_model1_{tag}.tsv", sep="\t", index=False)
    pkn_model2.to_csv(DATA_DIR / "network" / f"pkn_model2_{tag}.tsv", sep="\t", index=False)
    pkn_model3.to_csv(DATA_DIR / "network" / f"pkn_model3_{tag}.tsv", sep="\t", index=False)

    activities_early.to_csv(DATA_DIR / "differential" / f"activities_early_{tag}.tsv", sep="\t", index=False)
    secretome_early.to_csv(DATA_DIR / "differential" / f"secretome_early_{tag}.tsv", sep="\t", index=False)
    activities_late.to_csv(DATA_DIR / "differential" / f"activities_late_{tag}.tsv", sep="\t", index=False)

    print("\nSaved prepared inputs:")
    print(f"  {DATA_DIR / 'network' / f'pkn_model1_{tag}.tsv'}")
    print(f"  {DATA_DIR / 'network' / f'pkn_model2_{tag}.tsv'}")
    print(f"  {DATA_DIR / 'network' / f'pkn_model3_{tag}.tsv'}")
    print(f"  {DATA_DIR / 'differential' / f'activities_early_{tag}.tsv'}")
    print(f"  {DATA_DIR / 'differential' / f'secretome_early_{tag}.tsv'}")
    print(f"  {DATA_DIR / 'differential' / f'activities_late_{tag}.tsv'}")

    summary_rows.append({
        "feature_set": feature_set,
        "tag": tag,
        "n_activities_early": int(len(activities_early)),
        "n_secretome_early": int(len(secretome_early)),
        "n_activities_late": int(len(activities_late)),
        "pkn_model1_edges": int(len(pkn_model1)),
        "pkn_model2_edges": int(len(pkn_model2)),
        "pkn_model3_edges": int(len(pkn_model3)),
        "pkn_model1_nodes": int(len(pkn_nodes_m1)),
        "pkn_model2_nodes": int(len(pkn_nodes_m2)),
        "pkn_model3_nodes": int(len(pkn_nodes_m3)),
    })

# Summary table
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "inputs" / "feature_set_summary.tsv", sep="\t", index=False)
print("\nWrote:", RESULTS_DIR / "inputs" / "feature_set_summary.tsv")
print(summary_df.to_string(index=False))