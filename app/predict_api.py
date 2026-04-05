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
import numpy as np
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MODEL  = None
_GRAPH  = None
_OD     = None


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


def predict(u, v):
    model, G, od = load_artifacts()

    if not G.has_edge(u, v):
        return {"error": f"Edge {u}->{v} does not exist"}

    data     = G[u][v]
    capacity = data.get("capacity", 20)
    fft      = data.get("weight", 30.0)
    traffic  = np.random.uniform(0.5, 1.5)

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

    features = np.array([[traffic, capacity, fft, cent, alts, is_crit, 1.0]])

    try:
        impact = float(model.predict(features)[0])
    except Exception:
        # Fallback for v1 model (5 features)
        features_v1 = np.array([[traffic, capacity, fft, cent, alts]])
        impact = float(model.predict(features_v1)[0])

    impact = max(impact, 0.0)

    if impact < 0.1:
        recommendation = "Not critical — can be repurposed"
    elif impact < 0.3:
        recommendation = "Moderate importance — requires analysis"
    else:
        recommendation = "Critical — changes not recommended"

    return {
        "edge":           f"{u} -> {v}",
        "impact":         round(impact, 4),
        "impact_percent": round(impact * 100, 1),
        "recommendation": recommendation,
        "is_bridge":      bool(is_crit),
        "centrality":     round(cent, 4),
        "alternatives":   alts,
        "traffic":        round(traffic, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Road impact CLI")
    parser.add_argument("--from", dest="u", type=int)
    parser.add_argument("--to",   dest="v", type=int)
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
    print(f"Recommendation: {result['recommendation']}")
    print(f"Bridge: {result['is_bridge']}")
    print(f"Centrality: {result['centrality']}")
    print(f"Alternatives: {result['alternatives']}")


if __name__ == "__main__":
    main()
