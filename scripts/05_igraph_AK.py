from requests.packages import target
import igraph as ig
import matplotlib.pyplot as plt
import omnipath as op
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


##### This script builds an igraph graph from the CORNET results obtained prior #####

# error = some vertices in the edge DataFrame are missing from vertices dataframe > nodes in edges_df are missing in nodes_df 
# CORNETO output only stores nodes that have non-zero signal value in solution, but edged_df has more nodes in total (the extra compared to the nodes_df ere included structurally but were assigned no values (as pass-through nodes))
# missing nodes: 
edge_nodes = set(edges_df["source"]) | set(edges_df["target"])
vertex_nodes = set(nodes_df["node"])
missing_nodes = edge_nodes - vertex_nodes #27 nodes missing in nodes_df
print(f"Missing nodes in vertices DataFrame: {missing_nodes}")

#add missing todes with neutral defaults
missing_nodes_df = pd.DataFrame({
    "node": list(missing_nodes),
    "value": 0.0, # neutral default value
    "type": "intermediate" 
})
nodes_df_complete = pd.concat([nodes_df, missing_nodes_df], ignore_index=True)

#Build graph
g = ig.Graph.DataFrame(
    edges_df[["source", "target", "sign", "edge_value"]],
    directed=True,
    vertices=nodes_df_complete[["node", "value", "type"]],
    use_vids=False,
)

fig, ax = plt.subplots(figsize=(12, 12))
ig.plot(
    g,
    target=ax,
    vertex_label=g.vs["name"],
    vertex_size=20,
    edge_arrow_size=0.5,
    vertex_color=[
        "#ff6b6b" if v["type"] == "input" and v["name"] == "TGFB1"
        else "#ffb3b3" if v["type"] == "input"
        else "#b3e6b3" if v["type"] == "output"
        else "#f0f0f0"
        for v in g.vs
    ],
    edge_color=[
        "red" if e["edge_value"] > 0 else "blue"
        for e in g.es
    ],
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "network_IGRAPH.png", dpi=150, bbox_inches="tight")
plt.show()

# Remove isolated nodes (degree 0) — induced_subgraph preserves all edge/vertex attributes
vertices_to_keep = [v.index for v in g.vs if g.degree(v.index) > 0]
g_simplified = g.induced_subgraph(vertices_to_keep)

# Plot simplified graph
fig, ax = plt.subplots(figsize=(12, 12))
ig.plot(
    g_simplified,
    target=ax,
    vertex_label=g_simplified.vs["name"],
    vertex_size=20,
    edge_arrow_size=0.5,
    vertex_color=[
        "#ff6b6b" if v["type"] == "input" and v["name"] == "TGFB1"
        else "#ffb3b3" if v["type"] == "input"
        else "#b3e6b3" if v["type"] == "output"
        else "#f0f0f0"
        for v in g_simplified.vs
    ],
    edge_color=[
        "red" if e["edge_value"] > 0 else "blue"
        for e in g_simplified.es
    ],
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "network_IGRAPH_simplified.png", dpi=150, bbox_inches="tight")
plt.show()

#### basic network statistics ####
# %% 2. Basic network statistics
print("=== Basic Network Statistics ===")
print(f"Vertices     : {g_simplified.vcount()}") #190 nodes
print(f"Edges        : {g_simplified.ecount()}") #115
print(f"Density      : {g_simplified.density():.4f}") #0.0032
print(f"Diameter     : {g_simplified.diameter(directed=True)}") # 3 
print(f"Is connected : {g_simplified.is_connected(mode='weak')}") # FALSE
print(f"Components   : {len(g_simplified.connected_components(mode='weak'))}") #75

# Giant component stats
giant = g.connected_components(mode="weak").giant()
print(f"\nGiant component:")
print(f"  Nodes      : {giant.vcount()}")
print(f"  Edges      : {giant.ecount()}")
print(f"  Avg path   : {giant.average_path_length(directed=False):.3f}")
print(f"  Clustering : {giant.transitivity_undirected():.3f}")