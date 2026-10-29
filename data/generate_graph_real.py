import os
import pickle
import networkx as nx

try:
    import osmnx as ox
except ImportError:
    print("osmnx not installed. Run: pip install osmnx")
    exit(1)

CITY = "Astana, Kazakhstan"
SAVE_PATH = "data/graph_real.pkl"

# Controls how many nodes to keep.
# More nodes = more realistic map, but dataset generation takes longer.
#   500  nodes -> ~50-100 edges  -> dataset in ~5 min
#   1000 nodes -> ~100-200 edges -> dataset in ~15 min
#   2000 nodes -> ~200-400 edges -> dataset in ~40 min
SUBGRAPH_NODES = 500

PRESERVED_EDGE_ATTRS = (
    "osmid",
    "name",
    "highway",
    "oneway",
    "reversed",
    "length",
    "geometry",
    "maxspeed",
    "lanes",
    "bridge",
    "junction",
    "ref",
)


def pick_parallel_edge(raw_edges):
    """
    OSMnx MultiGraphs can contain several parallel OSM ways between the same
    nodes. Keep the shortest one so a NetworkX Graph still has one stable
    representative edge with real OSM geometry and metadata.
    """
    raw = dict(raw_edges)
    if not raw:
        return {}

    if not isinstance(next(iter(raw.values())), dict):
        return raw

    def length_of(attrs):
        try:
            return float(attrs.get("length", float("inf")))
        except Exception:
            return float("inf")

    return min(raw.values(), key=length_of)


def first_value(value, fallback=None):
    if isinstance(value, list):
        return value[0] if value else fallback
    return fallback if value is None else value


def parse_speed_kph(value, fallback=50.0):
    value = first_value(value, fallback)
    try:
        return float(str(value).split()[0])
    except Exception:
        return fallback


def parse_lanes(value, fallback=1):
    value = first_value(value, fallback)
    try:
        return max(int(float(str(value).split(";")[0])), 1)
    except Exception:
        return fallback


def build_edge_attrs(raw):
    attrs = {key: raw[key] for key in PRESERVED_EDGE_ATTRS if key in raw}

    travel_time = raw.get("travel_time")
    if travel_time is None:
        length = raw.get("length", 100)
        speed_kph = parse_speed_kph(raw.get("maxspeed"), fallback=50.0)
        try:
            travel_time = float(length) / (speed_kph * 1000 / 3600)
        except Exception:
            travel_time = 30.0

    try:
        attrs["weight"] = float(travel_time)
    except Exception:
        attrs["weight"] = 30.0

    lanes = parse_lanes(raw.get("lanes"), fallback=1)
    attrs["capacity"] = lanes * 20

    return attrs


def download_graph(city=CITY, subgraph_nodes=SUBGRAPH_NODES):
    print(f"Downloading road graph: {city}")

    G_directed = ox.graph_from_place(city, network_type="drive")
    G_directed = ox.add_edge_speeds(G_directed)
    G_directed = ox.add_edge_travel_times(G_directed)

    G = ox.convert.to_undirected(G_directed)
    print(f"Downloaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Take a geographically contiguous slice by starting from the node nearest
    # to the city center and expanding with BFS. This avoids grabbing an
    # arbitrary ordered prefix from OSMnx internals.
    center_lat, center_lon = 51.1694, 71.4491
    center_node = ox.distance.nearest_nodes(G, center_lon, center_lat)
    bfs_nodes = list(nx.bfs_tree(G, center_node, depth_limit=20).nodes())
    nodes = bfs_nodes[:subgraph_nodes]
    G_view = G.subgraph(nodes)
    largest_cc = max(nx.connected_components(G_view), key=len)

    # Build a new writable graph with OSM metadata preserved.
    G_new = nx.Graph()
    for node in largest_cc:
        G_new.add_node(node, **G.nodes[node])

    for u, v in G_view.subgraph(largest_cc).edges():
        raw = pick_parallel_edge(G_view[u][v])
        G_new.add_edge(u, v, **build_edge_attrs(raw))

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
