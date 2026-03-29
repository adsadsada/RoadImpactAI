import matplotlib.pyplot as plt
import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_graph import load_graph

def visualize_graph(save_path="docs/road_graph.png"):
    """Визуализирует граф дорог"""
    G = load_graph()
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    # узлы
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # рёбра
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5)
    
    # веса
    edge_labels = {(u, v): f"{d['weight']}мин" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Дорожная сеть города", fontsize=14)
    plt.axis('off')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Граф сохранён в {save_path}")
    plt.show()

if __name__ == "__main__":
    visualize_graph()