import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Road Optimizer", layout="wide")

st.title("🚦 AI Road Impact Analyzer")
st.write("Оцени влияние удаления дороги на городской трафик")

# -----------------------------
# Создание графа
# -----------------------------
@st.cache_data
def create_graph():
    G = nx.erdos_renyi_graph(n=10, p=0.4, seed=42)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)
    return G

G = create_graph()

# -----------------------------
# Функции
# -----------------------------
def avg_shortest_path(graph):
    lengths = []
    for node in graph.nodes():
        sp = nx.single_source_dijkstra_path_length(graph, node)
        lengths.extend(sp.values())
    return np.mean(lengths)

def generate_data(graph):
    data = []
    for (u, v) in list(graph.edges()):
        if graph.number_of_edges() < 2:
            continue

        base_time = avg_shortest_path(graph)

        traffic = np.random.uniform(0.5, 1.5)
        capacity = np.random.uniform(0.5, 1.5)
        centrality = nx.edge_betweenness_centrality(graph)[(u, v)]
        alternatives = len(list(nx.all_simple_paths(graph, u, v, cutoff=3)))

        graph.remove_edge(u, v)

        try:
            new_time = avg_shortest_path(graph)
        except:
            graph.add_edge(u, v, weight=np.random.randint(1, 10))
            continue

        graph.add_edge(u, v, weight=np.random.randint(1, 10))

        impact = (new_time - base_time) / base_time

        data.append([traffic, capacity, centrality, alternatives, base_time, impact])

    df = pd.DataFrame(data, columns=[
        'traffic', 'capacity', 'centrality',
        'alternatives', 'base_time', 'impact'
    ])
    return df

# -----------------------------
# Обучение модели
# -----------------------------
@st.cache_resource
def train_model(graph):
    df = generate_data(graph)
    X = df.drop(columns=['impact'])
    y = df['impact']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model, X.columns

model, feature_names = train_model(G)

# -----------------------------
# UI: выбор дороги
# -----------------------------
edges = list(G.edges())
edge_labels = [f"{u} - {v}" for u, v in edges]

selected_edge = st.selectbox("Выбери дорогу:", edge_labels)

u, v = edges[edge_labels.index(selected_edge)]

# -----------------------------
# Кнопка симуляции
# -----------------------------
if st.button("🔍 Predict Impact"):
    base_time = avg_shortest_path(G)

    traffic = np.random.uniform(0.5, 1.5)
    capacity = np.random.uniform(0.5, 1.5)
    centrality = nx.edge_betweenness_centrality(G)[(u, v)]
    alternatives = len(list(nx.all_simple_paths(G, u, v, cutoff=3)))

    sample = pd.DataFrame([[traffic, capacity, centrality, alternatives, base_time]],
                          columns=feature_names)

    prediction = model.predict(sample)[0]

    st.subheader("📊 Результат")

    st.metric("Impact Score", f"{prediction*100:.2f}%")

    if prediction > 0.2:
        st.error("❌ Дорога критична — удаление ухудшит трафик")
    else:
        st.success("✅ Дорога не критична — можно перепрофилировать")

    # -----------------------------
    # Explainability
    # -----------------------------
    st.subheader("🧠 Почему так?")
    importances = model.feature_importances_

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances)
    ax.set_title("Важность признаков")

    st.pyplot(fig)

# -----------------------------
# Визуализация графа
# -----------------------------
st.subheader("🗺️ Дорожная сеть")

fig, ax = plt.subplots()
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', ax=ax)

st.pyplot(fig)
