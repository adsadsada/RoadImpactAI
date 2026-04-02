import networkx as nx
import numpy as np
import pickle
import os

def create_city_graph(n_nodes=15, seed=42):
    """
    Создаёт реалистичный граф города с узкими местами.
    Используем планарный граф с низкой связностью,
    чтобы некоторые дороги были действительно критичными.
    """
    np.random.seed(seed)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Создаём структуру "районов" соединённых мостами
    # Район 1: узлы 0-4 (центр)
    center = [0, 1, 2, 3, 4]
    center_edges = [(0,1), (1,2), (2,3), (3,4), (0,3), (1,4)]

    # Район 2: узлы 5-8 (север)
    north = [5, 6, 7, 8]
    north_edges = [(5,6), (6,7), (7,8), (5,8)]

    # Район 3: узлы 9-11 (юг)
    south = [9, 10, 11]
    south_edges = [(9,10), (10,11)]

    # Район 4: узлы 12-14 (восток)
    east = [12, 13, 14]
    east_edges = [(12,13), (13,14), (12,14)]

    # Мосты между районами (критичные дороги!)
    bridges = [
        (1, 5),   # центр — север (единственный мост)
        (3, 9),   # центр — юг (единственный мост)
        (4, 12),  # центр — восток
        (2, 13),  # центр — восток (второй путь)
        (8, 14),  # север — восток (единственный мост)
        (11, 7),  # юг — север
    ]

    all_edges = center_edges + north_edges + south_edges + east_edges + bridges

    for u, v in all_edges:
        base_weight = np.random.uniform(2, 8)
        capacity = np.random.randint(20, 80)
        G.add_edge(u, v, weight=round(base_weight, 1), capacity=capacity)

    # Проверяем связность
    if not nx.is_connected(G):
        # Добавляем минимальные рёбра для связности
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i+1])[0]
            G.add_edge(u, v, weight=round(np.random.uniform(3, 7), 1), capacity=30)

    return G

def save_graph(graph, path="data/graph.pkl"):
    """Сохраняет граф в файл"""
    with open(path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"✅ Граф сохранён в {path}")

def load_graph(path="data/graph.pkl"):
    """Загружает граф из файла"""
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    G = create_city_graph(n_nodes=15)

    print(f"📊 Граф создан:")
    print(f"   - Узлов (перекрёстков): {G.number_of_nodes()}")
    print(f"   - Рёбер (дорог): {G.number_of_edges()}")
    print(f"   - Мосты (критичные): {len(list(nx.bridges(G)))}")

    print(f"\n📋 Примеры дорог:")
    edges_list = list(G.edges(data=True))
    for i, (u, v, data) in enumerate(edges_list[:5]):
        print(f"   {u} → {v}: {data['weight']} мин, пропускная способность: {data['capacity']}")

    save_graph(G)
