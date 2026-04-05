import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAVE_PATH = "docs/road_graph.png"


def visualize(save_path=SAVE_PATH):
    if os.path.exists("data/graph_real.pkl"):
        with open("data/graph_real.pkl", "rb") as f:
            G = pickle.load(f)
        title = "Road Network (OpenStreetMap)"
        # Use geographic coordinates if available
        if all("x" in G.nodes[n] and "y" in G.nodes[n] for n in G.nodes()):
            pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=42)
    else:
        from data.generate_graph import load_graph
        G = load_graph()
        title = "Road Network (Synthetic)"
        pos = nx.spring_layout(G, seed=42, k=1.5)

    bridges = set(nx.bridges(G))
    bridge_edges  = [(u, v) for u, v in G.edges() if (u, v) in bridges or (v, u) in bridges]
    regular_edges = [(u, v) for u, v in G.edges() if (u, v) not in bridges and (v, u) not in bridges]

    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    nx.draw_networkx_nodes(G, pos, node_color="#AED6F1", node_size=30, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, edge_color="#808080", width=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=bridge_edges,  edge_color="#C0392B", width=2.5, ax=ax)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    visualize()
