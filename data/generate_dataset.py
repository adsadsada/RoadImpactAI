import networkx as nx
import numpy as np
import pandas as pd
import sys
import os

# добавляем путь к корневой папке проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_graph import load_graph

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
                continue
    
    return total_time / total_trips if total_trips > 0 else 0

def calculate_edge_features(graph, edge, od_matrix):
    """Рассчитывает признаки для дороги"""
    u, v = edge
    
    traffic = np.random.uniform(0.3, 1.8)
    capacity = graph[u][v].get('capacity', 50)
    free_flow_time = graph[u][v]['weight']
    
    centrality = nx.edge_betweenness_centrality(graph)[(u, v)]
    
    try:
        alternatives = len(list(nx.all_simple_paths(graph, u, v, cutoff=3)))
        alternatives = min(alternatives, 10)
    except:
        alternatives = 1
    
    return {
        'traffic': traffic,
        'capacity': capacity,
        'free_flow_time': free_flow_time,
        'centrality': centrality,
        'alternatives': alternatives
    }

def simulate_edge_removal(graph, edge, od_matrix):
    """Симулирует удаление дороги и возвращает impact"""
    u, v = edge
    
    base_time = compute_avg_travel_time_with_demand(graph, od_matrix)
    
    weight_backup = graph[u][v]['weight']
    capacity_backup = graph[u][v].get('capacity', 50)
    
    graph.remove_edge(u, v)
    
    if nx.is_connected(graph):
        new_time = compute_avg_travel_time_with_demand(graph, od_matrix)
        impact = (new_time - base_time) / base_time if base_time > 0 else 0
    else:
        impact = 1.0
    
    graph.add_edge(u, v, weight=weight_backup, capacity=capacity_backup)
    
    return impact

def generate_dataset(graph, od_matrix, n_samples_per_edge=10):
    """Генерирует датасет для обучения"""
    data = []
    edges = list(graph.edges())
    
    for edge in edges:
        for _ in range(n_samples_per_edge):
            time_factor = np.random.choice([0.5, 1.0, 1.5])
            
            features = calculate_edge_features(graph, edge, od_matrix)
            features['traffic'] *= time_factor
            
            impact = simulate_edge_removal(graph, edge, od_matrix)
            
            data.append({
                'traffic': features['traffic'],
                'capacity': features['capacity'],
                'free_flow_time': features['free_flow_time'],
                'centrality': features['centrality'],
                'alternatives': features['alternatives'],
                'impact': impact
            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("📊 Загрузка графа и OD-матрицы...")
    G = load_graph()
    od = np.load("od_matrix.npy")
    
    print(f"Граф: {G.number_of_nodes()} узлов, {G.number_of_edges()} рёбер")
    print(f"OD-матрица: {od.shape}, всего поездок: {int(od.sum())}")
    
    print("🔄 Генерация датасета...")
    df = generate_dataset(G, od, n_samples_per_edge=10)
    
    print(f"✅ Датасет создан: {len(df)} записей")
    print("\n📋 Первые 5 строк:")
    print(df.head())
    
    df.to_csv("dataset.csv", index=False)
    print("\n💾 Сохранён в data/dataset.csv")