import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAPH_PATH = "data/graph_real.pkl"
OD_PATH    = "data/od_matrix_real.npy"
SAVE_PATH  = "data/dataset_real.csv"
SAMPLES    = 10

TIME_OF_DAY = {
    "night":        0.3,
    "morning_peak": 1.8,
    "day":          0.9,
    "evening_peak": 2.0,
    "evening":      0.6,
}

# Number of random OD pairs used to approximate avg travel time.
# Full O(n^2) computation is too slow for graphs with 500+ nodes.
# 300 random samples gives a stable estimate in a fraction of the time.
SAMPLE_OD_PAIRS = 300


def apply_bpr_weights(graph, traffic_multipliers=None, alpha=0.15, beta=4.0):
    """
    Apply BPR function to edge weights.
    t = free_flow_time * (1 + alpha * (volume / capacity) ^ beta)
    """
    G = graph.copy()
    for u, v, data in G.edges(data=True):
        fft = data.get("weight", 30.0)
        cap = data.get("capacity", 20)
        q   = traffic_multipliers.get((u, v),
              traffic_multipliers.get((v, u), cap * 0.5)) \
              if traffic_multipliers else cap * 0.5
        G[u][v]["weight"] = round(fft * (1 + alpha * (q / cap) ** beta), 2)
    return G


def sample_avg_travel_time(graph, node_list, od, rng, n_pairs=SAMPLE_OD_PAIRS):
    """
    Estimate average travel time using random OD pair sampling.
    Much faster than computing all n^2 shortest paths.
    """
    n = len(node_list)
    total, trips = 0.0, 0

    # Draw random origin-destination pairs weighted by demand
    # Flatten the OD matrix into a probability distribution
    flat = od.flatten()
    flat_sum = flat.sum()
    if flat_sum == 0:
        return 0.0

    probs = flat / flat_sum
    indices = rng.choice(len(probs), size=n_pairs, p=probs)

    for idx in indices:
        i, j = divmod(idx, n)
        if i == j:
            continue
        demand = od[i][j]
        if demand == 0:
            continue
        try:
            pl = nx.shortest_path_length(
                graph, node_list[i], node_list[j], weight="weight"
            )
            total += pl * demand
            trips += demand
        except nx.NetworkXNoPath:
            total += 9999 * demand
            trips += demand

    return total / trips if trips > 0 else 0.0


def compute_edge_features(graph, u, v, bridges, centrality, traffic_level, time_coeff):
    """Compute feature vector for a single edge."""
    data        = graph[u][v]
    weight      = data.get("weight", 30.0)
    capacity    = data.get("capacity", 20)
    cent        = centrality.get((u, v), centrality.get((v, u), 0.0))
    is_critical = 1 if (u, v) in bridges or (v, u) in bridges else 0

    try:
        alts = len(list(nx.all_simple_paths(graph, u, v, cutoff=3)))
        alts = min(max(alts - 1, 0), 10)
    except Exception:
        alts = 0

    return {
        "traffic":        traffic_level * time_coeff,
        "capacity":       capacity,
        "free_flow_time": weight,
        "centrality":     cent,
        "alternatives":   alts,
        "is_critical":    is_critical,
        "time_coeff":     time_coeff,
    }


def simulate_removal(graph, u, v, od, node_list, traffic_multipliers,
                     bridges, base_time, rng):
    """
    Simulate edge removal and return impact score.
    Impact = relative increase in avg travel time * scaling factor.
    Bridge edges receive a minimum impact of 0.6.
    """
    loaded = apply_bpr_weights(graph, traffic_multipliers)
    loaded.remove_edge(u, v)

    if nx.is_connected(loaded):
        new_time = sample_avg_travel_time(loaded, node_list, od, rng)
        impact   = (new_time - base_time) / base_time * 5 if base_time > 0 else 0.0
        impact   = float(np.clip(impact, 0.0, 1.0))
    else:
        impact = 0.5

    if (u, v) in bridges or (v, u) in bridges:
        impact = max(impact, 0.6)

    return impact


def generate_dataset(graph, od, node_list, n_samples=SAMPLES):
    edges     = list(graph.edges())
    bridges   = set(nx.bridges(graph))
    time_list = list(TIME_OF_DAY.items())
    rng       = np.random.default_rng(seed=42)

    print(f"Nodes: {len(node_list)}")
    print(f"Edges: {len(edges)}")
    print(f"Bridges: {len(bridges)}")
    print(f"Samples per edge: {n_samples}")
    print(f"Total rows: ~{len(edges) * n_samples}")
    print(f"OD pairs sampled per simulation: {SAMPLE_OD_PAIRS}")

    print("Computing edge betweenness centrality...")
    centrality = nx.edge_betweenness_centrality(graph, weight="weight", normalized=True)

    print("Computing base travel time...")
    base_loaded = apply_bpr_weights(graph)
    base_time   = sample_avg_travel_time(base_loaded, node_list, od, rng)
    print(f"Base time estimate: {base_time:.2f} sec")

    rows = []
    for idx, (u, v) in enumerate(edges):
        cap = graph[u][v].get("capacity", 20)

        for s in range(n_samples):
            time_name, time_coeff = time_list[s % len(time_list)]
            traffic_level  = rng.uniform(0.2, 2.0)
            actual_traffic = traffic_level * cap * time_coeff

            traffic_mults = {
                (eu, ev): rng.uniform(0.3, 1.2) * graph[eu][ev].get("capacity", 20) * time_coeff
                for eu, ev in graph.edges()
            }
            traffic_mults[(u, v)] = actual_traffic

            features = compute_edge_features(
                graph, u, v, bridges, centrality, traffic_level, time_coeff
            )
            impact = simulate_removal(
                graph, u, v, od, node_list, traffic_mults, bridges, base_time, rng
            )

            rows.append({**features, "impact": impact})

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(edges)} edges")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    if not os.path.exists(GRAPH_PATH):
        print(f"Graph not found: {GRAPH_PATH}")
        print("Run: python data/generate_graph_real.py")
        sys.exit(1)

    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    node_list = list(G.nodes())
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if not os.path.exists(OD_PATH):
        print(f"OD matrix not found: {OD_PATH}")
        print("Run: python data/od_matrix.py --real")
        sys.exit(1)

    od = np.load(OD_PATH)
    print(f"OD matrix: {od.shape}, total trips: {int(od.sum())}")

    print("\nGenerating dataset...")
    df = generate_dataset(G, od, node_list, n_samples=SAMPLES)

    os.makedirs("data", exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)

    print(f"\nDataset: {len(df)} rows -> {SAVE_PATH}")
    print(f"Impact: min={df['impact'].min():.4f}  mean={df['impact'].mean():.4f}  max={df['impact'].max():.4f}")
    print(f"Critical (>0.6): {(df['impact'] > 0.6).sum()} rows")
    print("Next step: python model/train_model_real.py")
