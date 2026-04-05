import os
import pickle
import networkx as nx

try:
    import osmnx as ox
except ImportError:
    print("osmnx not installed. Run: pip install osmnx")
    exit(1)

CITY      = "Almaty, Kazakhstan"
SAVE_PATH = "data/graph_real.pkl"

# Controls how many nodes to keep.
# More nodes = more realistic map, but dataset generation takes longer.
#   500  nodes -> ~50-100 edges  -> dataset in ~5 min
#   1000 nodes -> ~100-200 edges -> dataset in ~15 min
#   2000 nodes -> ~200-400 edges -> dataset in ~40 min
SUBGRAPH_NODES = 500


def download_graph(city=CITY, subgraph_nodes=SUBGRAPH_NODES):
    print(f"Downloading road graph: {city}")

    G_directed = ox.graph_from_place(city, network_type="drive")
    G_directed = ox.add_edge_speeds(G_directed)
    G_directed = ox.add_edge_travel_times(G_directed)

    G = ox.convert.to_undirected(G_directed)
    print(f"Downloaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Take a geographically contiguous slice — use the largest connected
    # component of the subgraph induced by the first N nodes in the node list.
    # Nodes in OSM graphs are ordered spatially, so this gives a real district.
    nodes = list(G.nodes())[:subgraph_nodes]
    G_view = G.subgraph(nodes)
    largest_cc = max(nx.connected_components(G_view), key=len)

    # Build a new writable graph — subgraph() returns a read-only view
    G_new = nx.Graph()
    for node in largest_cc:
        G_new.add_node(node, **G.nodes[node])

    for u, v in G_view.subgraph(largest_cc).edges():
        raw = dict(G_view[u][v])

        # MultiGraph stores parallel edges as nested dicts; take the first one
        if raw and isinstance(next(iter(raw.values())), dict):
            raw = next(iter(raw.values()))

        # weight = travel time in seconds
        tt = raw.get("travel_time", None)
        if tt is None:
            length = raw.get("length", 100)
            try:
                tt = float(length) / 13.9  # assume 50 km/h
            except Exception:
                tt = 30.0
        try:
            weight = float(tt)
        except Exception:
            weight = 30.0

        # capacity = lanes * 20 vehicles/min (rough approximation)
        lanes = raw.get("lanes", 1)
        if isinstance(lanes, list):
            lanes = lanes[0]
        try:
            capacity = int(lanes) * 20
        except Exception:
            capacity = 20

        G_new.add_edge(u, v, weight=weight, capacity=capacity)

    print(f"Subgraph: {G_new.number_of_nodes()} nodes, {G_new.number_of_edges()} edges")
    print(f"Bridges: {len(list(nx.bridges(G_new)))}")

    return G_new


def save_graph(G, path=SAVE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph_real(path=SAVE_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    G = download_graph()
    save_graph(G)
    print("Next step: python data/od_matrix.py --real")
