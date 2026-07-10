import argparse
import os
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from preprocessing import (
    ToxigenDataset,
    LGBTEnDataset,
    MigrantsEnDataset,
    MultilingualENCorpusGermanDataset,
    MultilingualENCorpusFrenchDataset,
    MultilingualENCorpusCypriotDataset,
    MultilingualENCorpusSloveneDataset,
)
from networkx.algorithms.community import girvan_newman

DATASETS = {
    "Toxigen": ToxigenDataset,
    "LGBTEn": LGBTEnDataset,
    "MigrantsEn": MigrantsEnDataset,
    "MultilingualENCorpusGerman": MultilingualENCorpusGermanDataset,
    "MultilingualENCorpusFrench": MultilingualENCorpusFrenchDataset,
    "MultilingualENCorpusCypriot": MultilingualENCorpusCypriotDataset,
    "MultilingualENCorpusSlovene": MultilingualENCorpusSloveneDataset,
}

def safe_fname(feature):
    return feature.replace("/", "_").replace(" ", "_")

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def edge_homophily(G, feature):
    same_group_edges = 0
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0
    for u, v in G.edges():
        if G.nodes[u].get(feature) == G.nodes[v].get(feature):
            same_group_edges += 1
    return same_group_edges / total_edges if total_edges > 0 else 0

def embed_image(img_path):
    return f"![{os.path.basename(img_path)}]({img_path})"

class GraphAnalysis:
    def __init__(self, adjacency_file, dataset, out_folder):
        with open(adjacency_file, 'rb') as f:
            adj_matrix = pickle.load(f)
        self.dataset = dataset
        self.node_features = self.dataset.data
        assert adj_matrix.shape[0] == len(self.node_features), "adjacency matrix and dataset size mismatch"
        if hasattr(adj_matrix, "numpy"):
            adj_matrix = adj_matrix.numpy()
        self.adj_matrix = adj_matrix
        self.out_folder = out_folder
        os.makedirs(self.out_folder, exist_ok=True)

    def create_nx_graph(self, threshold):
        adj_matrix = self.adj_matrix.copy()
        adj_matrix[adj_matrix < threshold] = 0
        np.fill_diagonal(adj_matrix, 0)
        G = nx.from_numpy_array(adj_matrix)
        nx.set_node_attributes(G, self.node_features.to_dict('index'))
        return G

    def summary_stats(self, threshold):
        G = self.create_nx_graph(threshold)
        stats = {
            "Nodes": G.number_of_nodes(),
            "Edges": G.number_of_edges(),
            "Connected Components": nx.number_connected_components(G),
            "Average Degree": np.mean([deg for _, deg in G.degree()]) if G.number_of_nodes() > 0 else 0,
        }
        summary = [
            f"Nodes: {stats['Nodes']}",
            f"Edges: {stats['Edges']}",
            f"Connected Components: {stats['Connected Components']}",
            f"Average Degree: {stats['Average Degree']:.2f}"
        ]
        if stats['Edges'] > stats['Nodes'] * 2:
            summary.append("Graph is dense (many connections per node).")
        elif stats['Edges'] < stats['Nodes']:
            summary.append("Graph is sparse (few connections per node).")
        else:
            summary.append("Graph has moderate connectivity.")
        if stats['Connected Components'] == 1:
            summary.append("All nodes are reachable; single connected component.")
        elif stats['Connected Components'] > 10:
            summary.append("Many clusters/components detected.")
        else:
            summary.append("Some separation into clusters/components.")
        summary.append("Tip: Adjust the threshold to tune density/separation.")
        return "\n".join(summary), stats, G

    def plot_connected_components(self, threshold):
        G = self.create_nx_graph(threshold)
        components = list(nx.connected_components(G))
        node_to_component = {node: idx for idx, comp in enumerate(components) for node in comp}
        colors = [node_to_component[node] for node in G.nodes()]
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.tab20, node_size=40)
        nx.draw_networkx_edges(G, pos, alpha=0.1)
        plt.title(f'Connected Components (threshold={threshold})')
        plt.axis('off')
        fname = os.path.join(self.out_folder, f"connected_components_{threshold}.png")
        plt.savefig(fname)
        plt.close()
        info = (
            f"Connected components plotted and saved to: {fname}\n"
            f"Each color is a cluster/component.\n"
            f"{embed_image(fname)}"
        )
        return info

    def plot_communities(self, threshold, nb_communities):
        G = self.create_nx_graph(threshold)
        communities_gen = girvan_newman(G)
        for communities in communities_gen:
            if len(communities) >= nb_communities:
                break
        node_labels = {}
        color_map = []
        for comm_idx, comm in enumerate(communities):
            for node in comm:
                node_labels[node] = str(comm_idx)
                color_map.append(comm_idx)
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 7))
        nx.draw_networkx_nodes(G, pos, node_color=color_map, cmap=plt.cm.tab20, node_size=40)
        nx.draw_networkx_edges(G, pos, alpha=0.1)
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        plt.title(f"Communities ({nb_communities}) at threshold={threshold}")
        plt.axis("off")
        fname = os.path.join(self.out_folder, f"communities_{nb_communities}_threshold{threshold}.png")
        plt.savefig(fname)
        plt.close()
        info = (
            f"Community structure plotted and saved to: {fname}\n"
            f"Each color/label is a detected community.\n"
            f"{embed_image(fname)}"
        )
        return info

    def save_group_ratios(self, threshold):
        G = self.create_nx_graph(threshold)
        results = []
        for idx, comp in enumerate(nx.connected_components(G)):
            nodes = list(comp)
            # Toxigen, LGBTEn, MigrantsEn might not have In-Group/Out-group, so skip if not present
            if "In-Group" in self.node_features.columns and "Out-group" in self.node_features.columns:
                comp_df = self.node_features.iloc[nodes][["In-Group", "Out-group"]]
                pair_counts = comp_df.value_counts()
                pair_ratios = pair_counts / pair_counts.sum()
                for (in_group, out_group), ratio in pair_ratios.items():
                    results.append({"Component": idx, "In-Group": in_group, "Out-group": out_group, "Ratio": ratio})
        info = ["Component-wise In-Group/Out-group ratios:"]
        for res in results:
            info.append(
                f"Component {res['Component']}: In-Group={res['In-Group']}, Out-group={res['Out-group']}, Ratio={res['Ratio']:.2f}"
            )
        return "\n".join(info)

    def homophily(self, threshold, feature):
        G = self.create_nx_graph(threshold)
        h = edge_homophily(G, feature)
        interp = ""
        if h > 0.8:
            interp = "High homophily: Most edges connect nodes with the SAME feature value."
        elif h > 0.5:
            interp = "Medium homophily: Many edges connect similar nodes, but there's also mixing."
        elif h > 0.2:
            interp = "Low homophily: Most edges connect nodes with DIFFERENT feature values."
        else:
            interp = "Very low homophily: Feature is NOT driving clustering."
        info = (
            f"Homophily for feature '{feature}': {h:.2f}\n"
            f"{interp}\n"
        )
        return info, {"Feature": feature, "Homophily": h, "Interpretation": interp}

    def homophily_all_columns(self, threshold, columns):
        G = self.create_nx_graph(threshold)
        infos = ["Homophily for all features:"]
        for feature in columns:
            h = edge_homophily(G, feature)
            if h > 0.8:
                interp = "High"
            elif h > 0.5:
                interp = "Medium"
            elif h > 0.2:
                interp = "Low"
            else:
                interp = "Very low"
            line = f"{feature}: {h:.2f} ({interp})"
            infos.append(line)
        return "\n".join(infos)

    def assortativity(self, threshold, feature):
        G = self.create_nx_graph(threshold)
        try:
            assort = nx.attribute_assortativity_coefficient(G, feature)
        except Exception:
            assort = np.nan
        interp = ""
        if not np.isnan(assort):
            if assort > 0.3:
                interp = "Nodes prefer connections to same attribute value (positive assortativity)."
            elif assort < -0.3:
                interp = "Nodes prefer connections to different attribute values (negative assortativity)."
            else:
                interp = "Random mixing (near zero assortativity)."
        else:
            interp = "Cannot compute assortativity."
        info = (
            f"Assortativity for feature '{feature}': {assort:.2f}\n"
            f"{interp}\n"
        )
        return info

class ThresholdAnalysis(GraphAnalysis):
    def __init__(self, adjacency_file, dataset, thresholds, out_folder):
        super().__init__(adjacency_file, dataset, out_folder)
        self.thresholds = thresholds

    def plot_nb_edges_variation(self):
        num_edges = []
        for t in self.thresholds:
            adj = self.adj_matrix.copy()
            adj[adj < t] = 0
            np.fill_diagonal(adj, 0)
            G = nx.from_numpy_array(adj)
            num_edges.append(G.number_of_edges())
        plt.figure(figsize=(8, 5))
        plt.plot(self.thresholds, num_edges)
        plt.xlabel("Threshold Value")
        plt.ylabel("Number of edges")
        plt.title("Number of edges depending on threshold")
        fname = os.path.join(self.out_folder, "edges_vs_threshold.png")
        plt.savefig(fname)
        plt.close()
        info = (
            "Number of edges vs threshold:\n"
            "Shows how increasing the threshold rapidly reduces the number of edges, sparsifying the network.\n"
            f"{embed_image(fname)}"
        )
        return info

    def plot_connected_components_evolution(self):
        num_components = []
        for t in self.thresholds:
            adj = self.adj_matrix.copy()
            adj[adj < t] = 0
            np.fill_diagonal(adj, 0)
            G = nx.from_numpy_array(adj)
            num_components.append(nx.number_connected_components(G))
        plt.figure(figsize=(8, 5))
        plt.plot(self.thresholds, num_components)
        plt.xlabel("Threshold Value")
        plt.ylabel("Number of connected components")
        plt.title("Number of connected components depending on threshold")
        fname = os.path.join(self.out_folder, "components_vs_threshold.png")
        plt.savefig(fname)
        plt.close()
        info = (
            "Number of connected components vs threshold:\n"
            "Shows how the graph fragments into more components as the threshold increases.\n"
            f"{embed_image(fname)}"
        )
        return info

    def plot_homophily_variation(self, feature):
        homophilies = []
        for t in self.thresholds:
            adj = self.adj_matrix.copy()
            adj[adj < t] = 0
            np.fill_diagonal(adj, 0)
            G = nx.from_numpy_array(adj)
            nx.set_node_attributes(G, self.node_features.to_dict('index'))
            h = edge_homophily(G, feature)
            homophilies.append(h)
        plt.figure(figsize=(8, 5))
        plt.plot(self.thresholds, homophilies, 'b.-')
        plt.xlabel("Threshold Value")
        plt.ylabel(f"Similarity Homophily")
        plt.title("Similarity homophily depending on edge threshold")
        fname = os.path.join(self.out_folder, f"{safe_fname(feature)}_homophily_vs_threshold.png")
        plt.savefig(fname)
        plt.close()
        info = (
            f"Similarity homophily vs threshold (feature: {feature}