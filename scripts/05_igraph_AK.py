import igraph as ig
import matplotlib.pyplot as plt
import pandas as pd


##### This script builds an igraph graph from the CORNET results obtained prior #####

### 1.0 network visualization with igraph ###
# error = some vertices in the edge DataFrame are missing from vertices dataframe > nodes in edges_df are missing in nodes_df 
# CORNETO output only stores nodes that have non-zero signal value in solution, but edged_df has more nodes in total (the extra compared to the nodes_df ere included structurally but were assigned no values (as pass-through nodes))
# missing nodes: 
edges_df, nodes_df = load_results("network", RESULTS_DIR)
print(f"Our network: {len(edges_df)} edges, {len(nodes_df)} nodes")


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
    vertex_size=[90 if v["name"] == "TGFB1" else 40 for v in g.vs],
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

# Remove isolated nodes (degree 0)
vertices_to_keep = [v.index for v in g.vs if g.degree(v.index) > 0]
g_simplified = g.induced_subgraph(vertices_to_keep)

# Plot simplified graph
fig, ax = plt.subplots(figsize=(12, 12))
ig.plot(
    g_simplified,
    target=ax,
    vertex_label=g_simplified.vs["name"],
    vertex_size=[90 if v["name"] == "TGFB1" else 40 for v in g.vs],
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

#### 2. basic network statistics ####
# %% 2. Basic network statistics
print("=== Basic Network Statistics ===")
print(f"Vertices     : {g_simplified.vcount()}") #173 nodes
print(f"Edges        : {g_simplified.ecount()}") #189
print(f"Density      : {g_simplified.density():.4f}") #0.0064
print(f"Diameter     : {g_simplified.diameter(directed=True)}") # 11
print(f"Is connected : {g_simplified.is_connected(mode='weak')}") # TRUE
print(f"Components   : {len(g_simplified.connected_components(mode='weak'))}") #1

# Giant component stats
# giant component = largest connected subgraph (biggest cluster of nodes)
giant = g_simplified.connected_components(mode="weak").giant()
print(f"\nGiant component:")
print(f"  Nodes      : {giant.vcount()}") #173
print(f"  Edges      : {giant.ecount()}") #189
print(f"  Avg path   : {giant.average_path_length(directed=False):.3f}") #6.011
print(f"  Clustering : {giant.transitivity_undirected():.3f}") #0.009


### hierarchical layout 
layout = g_simplified.layout_fruchterman_reingold()  # clusters nodes together
layout = g_simplified.layout_sugiyama()
fig, ax = plt.subplots(figsize=(16, 12))
ig.plot(
    g_simplified,
    target=ax,
    layout=layout,
    vertex_label=g_simplified.vs["name"],
    vertex_size=20,
    vertex_label_size=7,
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
plt.savefig(RESULTS_DIR / "network_IGRAPH_sugiyama.png", dpi=150, bbox_inches="tight")
plt.show()

#hierarchil layout of only the giant component
layout_giant = giant.layout_sugiyama()
fig, ax = plt.subplots(figsize=(12, 10))
ig.plot(
    giant,
    target=ax,
    layout=layout_giant,
    vertex_label=giant.vs["name"],
    vertex_size=20,
    vertex_label_size=7,
    edge_arrow_size=0.5,
    vertex_color=[
        "#ff6b6b" if v["type"] == "input" and v["name"] == "TGFB1"
        else "#ffb3b3" if v["type"] == "input"
        else "#b3e6b3" if v["type"] == "output"
        else "#f0f0f0"
        for v in giant.vs
    ],
    edge_color=[
        "red" if e["edge_value"] > 0 else "blue"
        for e in giant.es
    ],
)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "network_IGRAPH_giant_sugiyama.png", dpi=150, bbox_inches="tight")
plt.show()

### 3.0 node centrality (most important nodes in the network) ###
# centrality metrics on g_simplified 
degree_df = pd.DataFrame({
    "node": g_simplified.vs["name"],
    "type": g_simplified.vs["type"],
    "degree_in": g_simplified.degree(mode="in"),
    "degree_out": g_simplified.degree(mode="out"),
    "degree_total": g_simplified.degree(),
    "betweenness": g_simplified.betweenness(directed=True),
    "page_rank": g_simplified.pagerank(directed=True),
}) .sort_values("degree_total", ascending=False)

#top 15 nodes by betweenness centrality (potential bottlenecks)
top15 = degree_df.head(15)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# betweennes (controls flow, bottlenecks)
axes[0].barh(top15["node"], top15["betweenness"], color="#4c9be8")
axes[0].invert_yaxis()
axes[0].set_title("Top 15 — Betweenness Centrality")
axes[0].set_xlabel("Betweenness")

# PageRank (receives many incomig edges/signals)
top15_pr = degree_df.sort_values("page_rank", ascending=False).head(15)
axes[1].barh(top15_pr["node"], top15_pr["page_rank"], color="#f4a261")
axes[1].invert_yaxis()
axes[1].set_title("Top 15 — PageRank")
axes[1].set_xlabel("PageRank")

# Total degree (connected to many other nodes)
top15_deg = degree_df.sort_values("degree_total", ascending=False).head(15)
axes[2].barh(top15_deg["node"], top15_deg["degree_total"], color="#2a9d8f")
axes[2].invert_yaxis()
axes[2].set_title("Top 15 — Total Degree")
axes[2].set_xlabel("Degree")

#save plot
plt.tight_layout()
plt.savefig(RESULTS_DIR / "network_IGRAPH_centrality_top15.png", dpi=150, bbox_inches="tight")
plt.show()

# overlap of top nodes across metrics
top_betweenness = set(degree_df.sort_values("betweenness", ascending=False).head(15)["node"])
top_pagerank = set(degree_df.sort_values("page_rank", ascending=False).head(15)["node"])
top_degree = set(degree_df.sort_values("degree_total", ascending=False).head(15)["node"])

core_nodes = top_betweenness & top_pagerank & top_degree
print(f"Core nodes appearing in all 3 metrics: {core_nodes}") # MAPK1 & ATF3 \
#so many paths pass through them (bottlenecks), they receive many incoming signals, and they have many direct connections

# Pairwise overlaps
print(f"Betweenness ∩ PageRank  : {top_betweenness & top_pagerank}") #EMAPK1, ATF3, HIF1A
# betweennes: signal from many upstream nodes pass through these nodes 
# pagerank: receive many incoming edges/signals from other nodes
# >> so together they receive many upstream regulators and rellay the signal to many downstream nodes 
# == signal integrators/relayers 

print(f"Betweenness ∩ Degree    : {top_betweenness & top_degree}") #'TGFB1', 'CDK1', 'NFKB1', 'ATF3', 'MYC', 'TP53', 'AKT1', 'MAPK1', 'SP1', 'AR'
# degree: many direct connections
# betweennes: signal must pass through these nodes
## == core signal processors of TGF-β response  (bottlenecks)


print(f"PageRank    ∩ Degree    : {top_pagerank & top_degree}") #MAPK1, ATF3
# degree: many direct connections
# pagerank: many incoming edges/signals 
# == key downstream tartgets (so signal receivers/executors)

### community detection ###
# community_walktrap returns a VertexDendrogram; cut it into flat clusters with .as_clustering()
dendrogram = g_simplified.community_walktrap()
communities = dendrogram.as_clustering()  # cuts at the optimal modularity level
print(f"Number of communities: {len(communities)}")
print(f"Modularity score: {communities.modularity:.4f}")


# assign community membership to nodes
membership = communities.membership 
g_simplified.vs["community"] = membership

# assign a color per community
import random
random.seed(42)
n_communities = len(communities)
palette = ig.drawing.colors.ClusterColoringPalette(n_communities)
community_colors = [palette.get(m) for m in membership]

# plot with community colors
layout = g_simplified.layout_fruchterman_reingold()
fig, ax = plt.subplots(figsize=(16, 12))
ig.plot(
    g_simplified,
    target=ax,
    layout=layout,
    vertex_label=g_simplified.vs["name"],
    vertex_size=50,
    vertex_label_size=7,
    edge_arrow_size=0.5,
    vertex_color=community_colors,
    edge_color=[
        "red" if e["edge_value"] > 0 else "blue"
        for e in g_simplified.es
    ],
)
plt.title(f"Community Detection — Walktrap ({n_communities} communities, modularity={communities.modularity:.3f})")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "network_IGRAPH_communities.png", dpi=150, bbox_inches="tight")
plt.show()
