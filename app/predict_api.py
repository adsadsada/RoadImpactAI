"""
CLI tool for road impact prediction.

Usage:
    python app/predict_api.py --list-edges
    python app/predict_api.py --from 0 --to 1
"""

import sys
import os
import argparse
import pickle
import hashlib
import numpy as np
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MODEL = None
_GRAPH = None
_OD = None

TIME_COEFF = 1.3

IMPACT_WEIGHTS = {
    "model": 0.35,
    "traffic": 0.18,
    "centrality": 0.20,
    "alternatives": 0.12,
    "capacity": 0.07,
    "free_flow_time": 0.04,
    "bridge": 0.04,
}


def load_artifacts():
    global _MODEL, _GRAPH, _OD

    if _GRAPH is None:
        if os.path.exists("data/graph_real.pkl"):
            with open("data/graph_real.pkl", "rb") as f:
                _GRAPH = pickle.load(f)
        else:
            from data.generate_graph import load_graph
            _GRAPH = load_graph()
        print(f"Graph: {_GRAPH.number_of_nodes()} nodes, {_GRAPH.number_of_edges()} edges")

    if _MODEL is None:
        for path in [
            "model/random_forest_real.pkl",
            "model/random_forest_v2.pkl",
            "model/random_forest.pkl",
        ]:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    _MODEL = pickle.load(f)
                print(f"Model: {path}")
                break

    if _OD is None:
        for path in ["data/od_matrix_real.npy", "data/od_matrix.npy"]:
            if os.path.exists(path):
                _OD = np.load(path)
                print(f"OD matrix: {_OD.shape}")
                break

    return _MODEL, _GRAPH, _OD


def first_edge_data(graph, u, v):
    data = graph.get_edge_data(u, v, default={})
    if not isinstance(data, dict):
        return {}
    if any(key in data for key in (
        "geometry", "weight", "capacity", "length", "name", "osmid"
    )):
        return data
    for value in data.values():
        if isinstance(value, dict):
            return value
    return data


def normalize(value, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if high <= low:
        return 0.0
    return float(np.clip((number - low) / (high - low), 0.0, 1.0))


def clamp01(value):
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def stable_traffic_multiplier(u, v):
    key = f"{u}->{v}".encode("utf-8")
    seed = int(hashlib.sha256(key).hexdigest()[:16], 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.45, 1.65) * TIME_COEFF)


def calculate_adjusted_impact(model_impact, traffic, capacity, fft, cent, alts, is_crit):
    scores = {
        "model": clamp01(model_impact),
        "traffic": normalize(traffic, 0.40, 2.30),
        "centrality": normalize(cent, 0.00, 0.12),
        "alternatives": 1.0 - normalize(alts, 0.0, 10.0),
        "capacity": normalize(capacity, 20.0, 140.0),
        "free_flow_time": normalize(fft, 5.0, 120.0),
        "bridge": 1.0 if is_crit else 0.0,
    }
    weighted = sum(IMPACT_WEIGHTS[name] * scores[name] for name in IMPACT_WEIGHTS)
    return clamp01(weighted)


def predict(u, v):
    model, G, od = load_artifacts()

    if not G.has_edge(u, v):
        return {"error": f"Edge {u}->{v} does not exist"}

    data = first_edge_data(G, u, v)
    capacity = data.get("capacity", 20)
    fft = data.get("weight", 30.0)
    traffic = stable_traffic_multiplier(u, v)

    try:
        cent_dict = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
        cent = cent_dict.get((u, v), cent_dict.get((v, u), 0.0))
    except Exception:
        cent = 0.0

    try:
        alts = len(list(nx.all_simple_paths(G, u, v, cutoff=3)))
        alts = min(max(alts - 1, 0), 10)
    except Exception:
        alts = 0

    bridges = set(nx.bridges(G))
    is_crit = 1 if (u, v) in bridges or (v, u) in bridges else 0

    features = np.array([[traffic, capacity, fft, cent, alts, is_crit, TIME_COEFF]])

    try:
        model_impact = float(model.predict(features)[0])
    except Exception:
        features_v1 = np.array([[traffic, capacity, fft, cent, alts]])
        model_impact = float(model.predict(features_v1)[0])

    impact = calculate_adjusted_impact(
        model_impact, traffic, capacity, fft, cent, alts, bool(is_crit)
    )

    if impact < 0.18:
        recommendation = "Not critical - can be repurposed"
    elif impact < 0.38:
        recommendation = "Moderate importance - requires analysis"
    else:
        recommendation = "Critical - changes not recommended"

    return {
        "edge": f"{u} -> {v}",
        "impact": round(impact, 4),
        "impact_percent": round(impact * 100, 1),
        "model_impact": round(clamp01(model_impact), 4),
        "recommendation": recommendation,
        "is_bridge": bool(is_crit),
        "centrality": round(cent, 4),
        "alternatives": alts,
        "traffic": round(traffic, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Road impact CLI")
    parser.add_argument("--from", dest="u", type=int)
    parser.add_argument("--to", dest="v", type=int)
    parser.add_argument("--list-edges", action="store_true")
    args = parser.parse_args()

    _, G, _ = load_artifacts()

    if args.list_edges:
        print("\nAvailable edges:")
        for u, v in G.edges():
            print(f"  {u} -> {v}")
        return

    if args.u is None or args.v is None:
        print("Usage: python app/predict_api.py --from 0 --to 1")
        print("       python app/predict_api.py --list-edges")
        return

    result = predict(args.u, args.v)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    print(f"\nEdge: {result['edge']}")
    print(f"Impact: +{result['impact_percent']}%")
    print(f"Model estimate: {result['model_impact'] * 100:.1f}%")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Bridge: {result['is_bridge']}")
    print(f"Centrality: {result['centrality']}")
    print(f"Alternatives: {result['alternatives']}")
    print(f"Traffic: {result['traffic']}x")


if __name__ == "__main__":
    main()
