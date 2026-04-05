import networkx as nx
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_graph import load_graph

# Критические дороги (мосты между районами)
CRITICAL_EDGES = [(1,5), (3,9), (4,12), (2,13), (8,14), (11,7)]
CRITICAL_EDGES += [(v, u) for u, v in CRITICAL_EDGES]
CRITICAL_EDGES = list(set(CRITICAL_EDGES))

# Больше вариаций времени суток (5 коэффициентов)
TIME_OF_DAY_COEFFICIENTS = {
    'ночь':          0.3,   # 00:00–06:00 — минимальный трафик
    'утро_пик':      1.8,   # 07:00–09:00 — утренний час-пик
    'день':          0.9,   # 10:00–16:00 — дневной трафик
    'вечер_пик':     2.0,   # 17:00–19:00 — вечерний час-пик
    'вечер':         0.6,   # 20:00–23:00 — вечерний спад
}

def apply_bpr_weights(graph, traffic_multipliers=None, alpha=0.15, beta=4.0):
    """BPR-формула для расчёта реального времени проезда"""
    G = graph.copy()
    for u, v, data in G.edges(data=True):
        fft = data['weight']
        cap = data.get('capacity', 50)

        if traffic_multipliers and (u, v) in traffic_multipliers:
            traffic = traffic_multipliers[(u, v)]
        elif traffic_multipliers and (v, u) in traffic_multipliers:
            traffic = traffic_multipliers[(v, u)]
        else:
            traffic = cap * 0.5

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
                path_length = nx.shortest_path_length(graph, origin, destination, weight='weight')
                total_time += path_length * trips
                total_trips += trips
            except nx.NetworkXNoPath:
                total_time += 999 * trips
                total_trips += trips

    return total_time / total_trips if total_trips > 0 else 0

def calculate_edge_features_v2(graph, edge, od_matrix, traffic_level=None, time_coeff=1.0):
    """Расширенный расчёт признаков для дороги (v2)"""
    u, v = edge

    if traffic_level is None:
        traffic_level = np.random.uniform(0.3, 1.8)

    # Применяем коэффициент времени суток
    effective_traffic = traffic_level * time_coeff

    capacity    = graph[u][v].get('capacity', 50)
    free_flow   = graph[u][v]['weight']
    is_critical = 1 if (u, v) in CRITICAL_EDGES else 0

    centrality = nx.edge_betweenness_centrality(graph, weight='weight').get(
        (u, v), nx.edge_betweenness_centrality(graph, weight='weight').get((v, u), 0)
    )

    try:
        alternatives = len(list(nx.all_simple_paths(graph, u, v, cutoff=4)))
        alternatives = max(alternatives - 1, 0)
        alternatives = min(alternatives, 10)
    except Exception:
        alternatives = 0

    return {
        'traffic':        effective_traffic,
        'capacity':       capacity,
        'free_flow_time': free_flow,
        'centrality':     centrality,
        'alternatives':   alternatives,
        'is_critical':    is_critical,
        'time_coeff':     time_coeff,
    }

def simulate_edge_removal_v2(graph, edge, od_matrix, traffic_multipliers=None):
    """Симуляция удаления дороги — возвращает impact (v2)"""
    u, v = edge

    loaded_graph = apply_bpr_weights(graph, traffic_multipliers)
    base_time = compute_avg_travel_time_with_demand(loaded_graph, od_matrix)

    loaded_graph.remove_edge(u, v)

    if nx.is_connected(loaded_graph):
        new_time = compute_avg_travel_time_with_demand(loaded_graph, od_matrix)
        impact = (new_time - base_time) / base_time if base_time > 0 else 0
        impact = impact * 5
        impact = min(impact, 1.0)
    else:
        new_time = compute_avg_travel_time_with_demand(loaded_graph, od_matrix)
        impact = (new_time - base_time) / base_time if base_time > 0 else 1.0
        impact = max(impact, 0.5)

    # Критические дороги — минимум 0.6
    if (u, v) in CRITICAL_EDGES:
        impact = max(impact, 0.6)

    return impact

def generate_dataset_v2(graph, od_matrix, n_samples_per_edge=15):
    """Улучшенная генерация датасета (v2): 300+ записей, 5 периодов суток"""
    data = []
    edges = list(graph.edges())
    time_periods = list(TIME_OF_DAY_COEFFICIENTS.items())

    print(f"   Генерация v2: {len(edges)} рёбер x {n_samples_per_edge} сэмплов x {len(time_periods)} периодов")

    for idx, edge in enumerate(edges):
        u, v = edge
        cap = graph[u][v].get('capacity', 50)

        for sample in range(n_samples_per_edge):
            # Выбираем случайный период суток
            time_name, time_coeff = time_periods[sample % len(time_periods)]
            traffic_level = np.random.uniform(0.2, 2.0)
            actual_traffic = traffic_level * cap * time_coeff

            traffic_multipliers = {}
            for eu, ev in graph.edges():
                ecap = graph[eu][ev].get('capacity', 50)
                base_load = np.random.uniform(0.3, 1.2) * ecap * time_coeff
                traffic_multipliers[(eu, ev)] = base_load
            traffic_multipliers[(u, v)] = actual_traffic

            features = calculate_edge_features_v2(
                graph, edge, od_matrix, traffic_level, time_coeff
            )
            impact = simulate_edge_removal_v2(graph, edge, od_matrix, traffic_multipliers)

            data.append({
                'traffic':        features['traffic'],
                'capacity':       features['capacity'],
                'free_flow_time': features['free_flow_time'],
                'centrality':     features['centrality'],
                'alternatives':   features['alternatives'],
                'is_critical':    features['is_critical'],
                'time_coeff':     features['time_coeff'],
                'impact':         impact
            })

        if (idx + 1) % 5 == 0:
            print(f"   ... обработано {idx + 1}/{len(edges)} рёбер")

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("📊 Загрузка графа и OD-матрицы...")
    G  = load_graph()
    od = np.load("data/od_matrix.npy")

    print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    print(f"Критических дорог (мосты): {len(CRITICAL_EDGES) // 2}")
    print(f"OD-матрица: {od.shape}, всего поездок: {int(od.sum())}")

    print("\n🔄 Генерация датасета v2...")
    df = generate_dataset_v2(G, od, n_samples_per_edge=15)

    print(f"\n✅ Датасет v2 создан: {len(df)} записей")
    print(f"\n📋 Статистика impact:")
    print(f"   min:  {df['impact'].min():.4f}")
    print(f"   mean: {df['impact'].mean():.4f}")
    print(f"   max:  {df['impact'].max():.4f}")
    print(f"   > 0.1: {(df['impact'] > 0.1).sum()} записей")
    print(f"   > 0.6: {(df['impact'] > 0.6).sum()} записей (критические)")

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset_v2.csv", index=False)
    print("\n💾 Сохранён в data/dataset_v2.csv")
