import networkx as nx
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_graph import load_graph

# Список критических дорог (мостов)
CRITICAL_EDGES = [(1,5), (3,9), (4,12), (2,13), (8,14), (11,7)]
# Добавляем обратные рёбра (граф ненаправленный)
CRITICAL_EDGES += [(v, u) for u, v in CRITICAL_EDGES]
CRITICAL_EDGES = list(set(CRITICAL_EDGES))

def apply_bpr_weights(graph, traffic_multipliers=None, alpha=0.15, beta=4.0):
    """
    Применяет BPR-формулу для расчёта реального времени проезда:
    t = free_flow_time * (1 + alpha * (traffic / capacity) ^ beta)

    traffic_multipliers — словарь {(u,v): множитель загрузки}
    """
    G = graph.copy()
    for u, v, data in G.edges(data=True):
        fft = data['weight']
        cap = data.get('capacity', 50)

        if traffic_multipliers and (u, v) in traffic_multipliers:
            traffic = traffic_multipliers[(u, v)]
        elif traffic_multipliers and (v, u) in traffic_multipliers:
            traffic = traffic_multipliers[(v, u)]
        else:
            traffic = cap * 0.5  # по умолчанию 50% загрузки

        actual_time = fft * (1 + alpha * (traffic / cap) ** beta)
        G[u][v]['weight'] = round(actual_time, 2)

    return G

def compute_avg_travel_time_with_demand(graph, od_matrix):
    """Среднее время поездки с учётом спроса"""
    total_time = 0
    total_trips = 0

    for origin in graph.nodes():
        for destination in graph.nodes():
            if origin == destination:
                continue

            trips = od_matrix[origin][destination]
            if trips == 0:
                continue

            try:
                path_length = nx.shortest_path_length(
                    graph, origin, destination, weight='weight'
                )
                total_time += path_length * trips
                total_trips += trips
            except nx.NetworkXNoPath:
                # Если путь не существует — штраф
                total_time += 999 * trips
                total_trips += trips

    return total_time / total_trips if total_trips > 0 else 0

def calculate_edge_features(graph, edge, od_matrix, traffic_level=None):
    """Рассчитывает признаки для дороги"""
    u, v = edge

    # traffic_level — реальная загрузка (доля от capacity)
    if traffic_level is None:
        traffic_level = np.random.uniform(0.3, 1.8)

    capacity = graph[u][v].get('capacity', 50)
    free_flow_time = graph[u][v]['weight']

    centrality = nx.edge_betweenness_centrality(graph, weight='weight')[(u, v)]

    try:
        alternatives = len(list(nx.all_simple_paths(graph, u, v, cutoff=4)))
        alternatives = max(alternatives - 1, 0)  # не считаем прямой путь
        alternatives = min(alternatives, 10)
    except:
        alternatives = 0

    return {
        'traffic': traffic_level,
        'capacity': capacity,
        'free_flow_time': free_flow_time,
        'centrality': centrality,
        'alternatives': alternatives
    }

def simulate_edge_removal(graph, edge, od_matrix, traffic_multipliers=None):
    """
    Симулирует удаление дороги и возвращает impact.
    Учитывает текущую загрузку через BPR.
    """
    u, v = edge

    # Применяем BPR к базовому графу
    loaded_graph = apply_bpr_weights(graph, traffic_multipliers)
    base_time = compute_avg_travel_time_with_demand(loaded_graph, od_matrix)

    # Удаляем ребро
    loaded_graph.remove_edge(u, v)

    if nx.is_connected(loaded_graph):
        new_time = compute_avg_travel_time_with_demand(loaded_graph, od_matrix)
        impact = (new_time - base_time) / base_time if base_time > 0 else 0
        impact = impact * 5   # УМНОЖАЕМ НА 5, чтобы сделать различия заметнее
        impact = min(impact, 1.0)  # ограничиваем максимум 100%
    else:
        # Граф разорван — очень высокий impact
        new_time = compute_avg_travel_time_with_demand(loaded_graph, od_matrix)
        impact = (new_time - base_time) / base_time if base_time > 0 else 1.0
        impact = max(impact, 0.5)  # минимум 50% если граф разорвался
    # Принудительно делаем критические дороги очень важными
    if (u, v) in CRITICAL_EDGES:
        impact = max(impact, 0.6)  # минимум 60% impact
    return impact

def generate_dataset(graph, od_matrix, n_samples_per_edge=10):
    """Генерирует датасет для обучения"""
    data = []
    edges = list(graph.edges())

    print(f"   Генерация для {len(edges)} рёбер x {n_samples_per_edge} сэмплов...")

    for idx, edge in enumerate(edges):
        u, v = edge
        cap = graph[u][v].get('capacity', 50)

        for sample in range(n_samples_per_edge):
            # Разные уровни загрузки
            traffic_level = np.random.uniform(0.2, 2.0)
            actual_traffic = traffic_level * cap

            # Создаём traffic_multipliers для всех рёбер
            traffic_multipliers = {}
            for eu, ev in graph.edges():
                ecap = graph[eu][ev].get('capacity', 50)
                # Базовая загрузка + шум
                base_load = np.random.uniform(0.3, 1.2) * ecap
                traffic_multipliers[(eu, ev)] = base_load

            # Для текущего ребра — конкретная загрузка
            traffic_multipliers[(u, v)] = actual_traffic

            features = calculate_edge_features(graph, edge, od_matrix, traffic_level)
            impact = simulate_edge_removal(graph, edge, od_matrix, traffic_multipliers)

            data.append({
                'traffic': features['traffic'],
                'capacity': features['capacity'],
                'free_flow_time': features['free_flow_time'],
                'centrality': features['centrality'],
                'alternatives': features['alternatives'],
                'impact': impact
            })

        if (idx + 1) % 5 == 0:
            print(f"   ... обработано {idx + 1}/{len(edges)} рёбер")

    return pd.DataFrame(data)

if __name__ == "__main__":
    print("📊 Загрузка графа и OD-матрицы...")
    G = load_graph()
    od = np.load("data/od_matrix.npy")

    print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    print(f"Мосты: {len(list(nx.bridges(G)))}")
    print(f"OD-матрица: {od.shape}, всего поездок: {int(od.sum())}")

    print("🔄 Генерация датасета...")
    df = generate_dataset(G, od, n_samples_per_edge=10)

    print(f"\n✅ Датасет создан: {len(df)} записей")
    print(f"\n📋 Статистика impact:")
    print(f"   min:  {df['impact'].min():.4f}")
    print(f"   mean: {df['impact'].mean():.4f}")
    print(f"   max:  {df['impact'].max():.4f}")
    print(f"   > 0.1: {(df['impact'] > 0.1).sum()} записей")
    print(f"   > 0.2: {(df['impact'] > 0.2).sum()} записей")

    print(f"\n📋 Первые 5 строк:")
    print(df.head())

    df.to_csv("data/dataset.csv", index=False)
    print("\n💾 Сохранён в data/dataset.csv")
