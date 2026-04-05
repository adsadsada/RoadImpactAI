import numpy as np
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAVE_PATH_SYNTH = "data/od_matrix.npy"
SAVE_PATH_REAL  = "data/od_matrix_real.npy"


def generate_od_matrix(graph, city_type="mixed"):
    """
    Generate an Origin-Destination demand matrix.

    city_type options:
      mixed       - uniform random demand between all node pairs
      center      - high demand between all nodes (dense urban core)
      residential - demand concentrated toward central nodes
      gravity     - gravity model based on node degree and shortest path distance
    """
    import networkx as nx

    n = graph.number_of_nodes()
    matrix = np.zeros((n, n))

    if city_type == "center":
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = np.random.randint(50, 200)

    elif city_type == "residential":
        center_nodes = list(range(min(3, n)))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if j in center_nodes and i not in center_nodes:
                    matrix[i][j] = np.random.randint(50, 150)
                elif i in center_nodes and j not in center_nodes:
                    matrix[i][j] = np.random.randint(30, 100)
                elif i not in center_nodes and j not in center_nodes:
                    matrix[i][j] = np.random.randint(5, 30)
                else:
                    matrix[i][j] = np.random.randint(10, 60)

    elif city_type == "gravity":
        # Trips between i and j proportional to degree(i) * degree(j) / distance(i,j)
        degrees = dict(graph.degree())
        nodes = list(graph.nodes())
        pop = np.array([degrees[node] + 1 for node in nodes], dtype=float)

        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                if i == j:
                    continue
                try:
                    dist = nx.shortest_path_length(graph, ni, nj, weight="weight")
                    dist = max(dist, 1.0)
                except Exception:
                    dist = 999.0
                matrix[i][j] = pop[i] * pop[j] / dist

        mx = matrix.max()
        if mx > 0:
            matrix = (matrix / mx * 190 + 10).astype(int)
        np.fill_diagonal(matrix, 0)

    else:  # mixed
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = np.random.randint(10, 100)

    return matrix.astype(int)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Use real OSM graph (graph_real.pkl)")
    parser.add_argument("--type", default=None,
                        help="Matrix type: center / residential / mixed / gravity")
    args = parser.parse_args()

    if args.real:
        from data.generate_graph_real import load_graph_real
        G    = load_graph_real()
        mode = args.type or "gravity"
        save = SAVE_PATH_REAL
    else:
        from data.generate_graph import load_graph
        G    = load_graph()
        mode = args.type or "mixed"
        save = SAVE_PATH_SYNTH

    print(f"Generating OD matrix (type: {mode}, nodes: {G.number_of_nodes()})...")
    od = generate_od_matrix(G, city_type=mode)

    print(f"Shape: {od.shape}")
    print(f"Total trips: {int(od.sum())}")
    print(f"Mean per route: {od[od > 0].mean():.1f}")

    os.makedirs("data", exist_ok=True)
    np.save(save, od)
    print(f"Saved to {save}")
