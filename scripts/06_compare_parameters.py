import pandas as pd
from pathlib import Path

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

network = pd.read_csv(RESULTS_DIR / "network_edges.tsv", sep="\t")
edge_tuples = list(zip(network["source"], network["mor"], network["target"]))
G = Graph.from_tuples(edge_tuples)

