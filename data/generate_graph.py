import networkx as nx
import numpy as np
import pickle
import os

def create_city_graph(n_nodes=15, seed=42):
    """
    Создаёт реалистичный граф города
    - n_nodes: количество перекрёстков
    """
    np.random.seed(seed)
    
    # создаём граф с разной связностью (как в реальном городе)
    G = nx.erdos_renyi_graph(n=n_nodes, p=0.35, seed=seed)
    
    # добавляем веса (время проезда в минутах)
    for u, v in G.edges():
        # разное время: центр быстрее, окраины медленнее
        base_weight = np.random.uniform(1, 5)
        G[u][v]['weight'] = round(base_weight, 1)
        G[u][v]['capacity'] = np.random.randint(30, 100)  # пропускная способность
    
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
    # создаём папку data если её нет
    os.makedirs("data", exist_ok=True)
    
    # создаём граф
    G = create_city_graph(n_nodes=15)
    
    print(f"📊 Граф создан:")
    print(f"   - Узлов (перекрёстков): {G.number_of_nodes()}")
    print(f"   - Рёбер (дорог): {G.number_of_edges()}")
    
    # выводим первые 5 дорог для примера
    print(f"\n📋 Примеры дорог:")
    edges_list = list(G.edges(data=True))
    for i, (u, v, data) in enumerate(edges_list[:5]):
        print(f"   {u} → {v}: {data['weight']} мин, пропускная способность: {data['capacity']}")
    
    # сохраняем
    save_graph(G)