import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Создаем граф дорог
# -----------------------------
G = nx.erdos_renyi_graph(n=10, p=0.4, seed=42, directed=False)

# добавляем веса (время проезда)
for u, v in G.edges():
    G[u][v]['weight'] = np.random.randint(1, 10)

# -----------------------------
# 2. Функция расчета среднего пути
# -----------------------------
def avg_shortest_path(graph):
    lengths = []
    for node in graph.nodes():
        sp = nx.single_source_dijkstra_path_length(graph, node)
        lengths.extend(sp.values())
    return np.mean(lengths)

# -----------------------------
# 3. Генерация датасета
# -----------------------------
data = []

for (u, v) in list(G.edges()):
    if G.number_of_edges() < 2:
        continue

    # baseline
    base_time = avg_shortest_path(G)

    # характеристики
    traffic = np.random.uniform(0.5, 1.5)
    capacity = np.random.uniform(0.5, 1.5)

    centrality = nx.edge_betweenness_centrality(G)[(u, v)]
    alternatives = len(list(nx.all_simple_paths(G, u, v, cutoff=3)))

    # удаляем дорогу
    G.remove_edge(u, v)

    try:
        new_time = avg_shortest_path(G)
    except:
        G.add_edge(u, v, weight=np.random.randint(1, 10))
        continue

    # возвращаем дорогу
    G.add_edge(u, v, weight=np.random.randint(1, 10))

    # impact
    impact = (new_time - base_time) / base_time

    data.append([
        traffic,
        capacity,
        centrality,
        alternatives,
        base_time,
        impact
    ])

# DataFrame
df = pd.DataFrame(data, columns=[
    'traffic', 'capacity', 'centrality',
    'alternatives', 'base_time', 'impact'
])

# -----------------------------
# 4. Обучение модели
# -----------------------------
X = df.drop(columns=['impact'])
y = df['impact']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# -----------------------------
# 5. Тестовое предсказание
# -----------------------------
sample = X_test.iloc[0:1]
prediction = model.predict(sample)

print("Prediction (impact):", prediction[0])
print("Real impact:", y_test.iloc[0])

# -----------------------------
# 6. Важность признаков
# -----------------------------
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)

