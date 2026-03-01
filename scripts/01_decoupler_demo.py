"""Transcription factor activity inference with Decoupler.

This script demonstrates how transcription factor (TF) activities can be
inferred from transcriptomics data. We use the Univariate Linear Model (ULM)
method from Decoupler together with TF-target regulons from CollecTRI.

The result is a matrix of TF activity scores across time points, showing
which TFs are activated or repressed upon TGF-beta stimulation.

The `print` statements help the inspection of the objects after each step, if
you run interactively, feel free to inspect them your own way.
"""

import pandas as pd
import decoupler as dc
import matplotlib.pyplot as plt
import seaborn as sns
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

# %% Load differential expression data

diff_expr = pd.read_csv(DATA_DIR / "differential" / "diff_expr_all.tsv", sep="\t")

print(f"Loaded {len(diff_expr)} rows")
print(f"Modalities: {diff_expr['modality'].unique()}")
print(f"Time points: {diff_expr['time'].unique()}")

# %% Filter for transcriptomics

rna = diff_expr[diff_expr["modality"] == "rna"].copy()
print(f"\nTranscriptomics: {len(rna)} measurements")
print(f"Genes: {rna['feature_id'].nunique()}")

# %% Pivot to a matrix: genes (rows) x time points (columns)
# Values are log fold changes (TGF-beta vs control)

rna_mat = rna.pivot_table(
    index="feature_id", columns="time", values="logFC", aggfunc="first"
)

# Order time points chronologically
time_order = ["0.08h", "1h", "12h", "24h", "48h", "72h", "96h"]
time_order = [t for t in time_order if t in rna_mat.columns]
rna_mat = rna_mat[time_order]

print(f"\nExpression matrix: {rna_mat.shape[0]} genes x {rna_mat.shape[1]} time points")
print(rna_mat.head())

# %% Get CollecTRI regulons
# CollecTRI is a comprehensive collection of TF-target interactions
# Decoupler accesses prior-knowledge from OmniPath

collectri = dc.op.collectri(organism="human")

print(f"\nCollecTRI regulons: {len(collectri)} interactions")
print(f"TFs: {collectri['source'].nunique()}")
print(f"Targets: {collectri['target'].nunique()}")
print(collectri.head())

# %% Run Univariate Linear Model (ULM)
# ULM fits a linear model for each TF, using the TF's targets' fold changes
# as observations and the mode of regulation (activation/inhibition) as the
# independent variable. The resulting t-statistic is the TF activity score.

acts, pvals = dc.mt.ulm(rna_mat.T, collectri)

print(f"\nTF activity scores: {acts.shape}")
print(
    "Number of significant TFs (p < 0.05 and |score| > 2) in "
    "each time point: \n\n"
    f"{((pvals < 0.05) & (acts.abs() > 2)).sum(axis = 1)}"
)

# %% Select TFs with significant activity in at least one time point
#
# acts has shape (time_points x TFs), so we check per-TF across time (axis=0)

sig_mask = (pvals < 0.05) & (acts.abs() > 2)
sig_tfs = sig_mask.any(axis=0)
acts_sig = acts.loc[:, sig_tfs].copy()

print(f"\nSignificant TFs (|score| > 2 and p < 0.05): {acts_sig.shape[1]}")

# %% Visualize: heatmap of TF activities across time
#
# Transpose so TFs are rows, time points are columns

# Sort TFs by their maximum absolute activity
max_act = acts_sig.abs().max(axis=0).sort_values(ascending=False)
top_tfs = max_act.index[:30]
# TFs as rows, time points as columns
acts_plot = acts_sig[top_tfs].T

fig, ax = plt.subplots(figsize=(8, 10))
sns.heatmap(
    acts_plot,
    cmap="RdBu_r",
    center=0,
    xticklabels=True,
    yticklabels=True,
    ax=ax,
    cbar_kws={"label": "TF activity (ULM score)"},
)
ax.set_title("TF activities upon TGF-β stimulation\n(top 30 by max |score|)")
ax.set_xlabel("Time after TGF-β stimulation")
ax.set_ylabel("Transcription factor")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "tf_activities_heatmap.pdf", bbox_inches="tight")
plt.savefig(RESULTS_DIR / "tf_activities_heatmap.png", dpi=150, bbox_inches="tight")
print(f"\nSaved heatmap to {RESULTS_DIR / 'tf_activities_heatmap.pdf'}")
plt.show()

# %% Save TF activity results

acts.to_csv(RESULTS_DIR / "tf_activity_scores.tsv", sep="\t")
pvals.to_csv(RESULTS_DIR / "tf_activity_pvalues.tsv", sep="\t")
print(f"Saved activity scores and p-values to {RESULTS_DIR}")

# %% Key observations
#
# Compare the activities with the ones in the paper
# There you can find interpretation, hypothesis and further validation for many
# Do you notice any difference?
# Here we used a completely different prior-knowledge of TF regulons: CollecTRI
# instead of DoRothEA
# Some starting points for interpretation:
# - SMAD2/3/4 are among the top activated TFs, consistent with canonical
#   TGF-beta signaling through the SMAD pathway
# - Several TFs show early activation (0.08h-1h) while others respond later
# - Some TFs are repressed (negative scores), suggesting active down-regulation
#
# These TF activities, together with kinase activities from phosphoproteomics,
# will serve as inputs for the network inference step with CORNETO.
