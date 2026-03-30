import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_graph import load_graph
from data.generate_dataset import compute_avg_travel_time_with_demand, calculate_edge_features

st.set_page_config(page_title="Road Impact AI", page_icon="🚦", layout="wide")

st.title("🚦 Road Impact AI")
st.markdown("### Система оценки влияния дорог на городской трафик")

@st.cache_resource
def load_artifacts():
    with open("model/random_forest.pkl", 'rb') as f:
        model = pickle.load(f)
    G = load_graph()
    importance = pd.read_csv("model/feature_importance.csv")
    return model, G, importance

try:
    model, G, importance = load_artifacts()
    st.success("✅ Модель и данные загружены")
except Exception as e:
    st.error(f"❌ Ошибка загрузки: {e}")
    st.stop()

@st.cache_resource
def get_od_matrix():
    n = G.number_of_nodes()
    od = np.random.randint(10, 100, size=(n, n))
    np.fill_diagonal(od, 0)
    return od

od_matrix = get_od_matrix()

with st.sidebar:
    st.header("⚙️ О системе")
    st.markdown("""
    **Как это работает:**
    1. Выберите дорогу на карте
    2. Система оценит её важность
    3. Получите рекомендацию
    """)
    st.divider()
    st.header("📊 О модели")
    st.write(f"**Алгоритм:** Random Forest Regressor")
    st.write(f"**Дорог в сети:** {G.number_of_edges()}")

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🗺️ Дорожная сеть города")
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42, k=1.5)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1.5, ax=ax)
    ax.set_title("Схема дорожной сети")
    ax.axis('off')
    st.pyplot(fig)
    
    edges = list(G.edges())
    edge_labels = [f"{u} → {v}" for u, v in edges]
    selected_idx = st.selectbox("Выберите дорогу для анализа:", range(len(edges)), format_func=lambda x: edge_labels[x])
    u, v = edges[selected_idx]

with col2:
    st.subheader("🔍 Анализ дороги")
    if st.button("🚀 Рассчитать влияние", type="primary"):
        with st.spinner("Анализируем дорогу..."):
            features = calculate_edge_features(G, (u, v), od_matrix)
            features_array = np.array([[
                features['traffic'], features['capacity'], 
                features['free_flow_time'], features['centrality'], 
                features['alternatives']
            ]])
            impact = model.predict(features_array)[0]
            impact_percent = impact * 100
            
            st.metric("🚗 Текущая загрузка", f"{features['traffic']:.1f}x")
            st.metric("📊 Центральность", f"{features['centrality']:.3f}")
            st.metric("🔄 Альтернативные пути", features['alternatives'])
            st.divider()
            
            if impact < 0.1:
                st.success(f"**Impact: +{impact_percent:.1f}%**")
                st.info("✅ **Рекомендация:** Дорога НЕ критична")
                st.markdown("**Предложения:**\n- 🌳 Пешеходная зона\n- 🚴 Велодорожка\n- 🌿 Сквер или парк")
            elif impact < 0.2:
                st.warning(f"**Impact: +{impact_percent:.1f}%**")
                st.info("⚠️ **Рекомендация:** Требует анализа")
                st.markdown("**Предложения:**\n- 📏 Сужение проезжей части\n- 🚶 Расширение тротуаров")
            else:
                st.error(f"**Impact: +{impact_percent:.1f}%**")
                st.info("❌ **Рекомендация:** Дорога критична")
                st.markdown("**Предложения:**\n- 🛣️ Оставить как есть\n- 🔧 Улучшить покрытие")
            
            st.divider()
            st.subheader("🧠 Почему такой вывод?")
            if features['centrality'] > 0.1:
                st.write(f"• Высокая центральность ({features['centrality']:.3f}) — дорога важна для связности сети")
            if features['alternatives'] < 3:
                st.write(f"• Мало альтернативных путей ({features['alternatives']})")
            if features['traffic'] > 0.8:
                st.write(f"• Высокая текущая загрузка ({features['traffic']:.1f}x)")
            if features['centrality'] <= 0.1 and features['alternatives'] >= 3:
                st.write("• Дорога имеет низкую центральность и достаточно альтернатив")

st.divider()
st.markdown("**🔬 Ограничения:** Модель обучена на симулированных данных. Требует калибровки на реальных данных.")