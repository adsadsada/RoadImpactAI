import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURE_COLS = [
    "traffic", "capacity", "free_flow_time",
    "centrality", "alternatives", "is_critical", "time_coeff",
]

# Fixed time coefficient — peak hour average.
# Time-of-day selection was removed: a road that is critical at peak hour
# cannot be repurposed just because it is quiet at night.
TIME_COEFF = 1.3


def make_node_labels(node_list):
    """
    Assign short alphabetic labels to nodes: a, b, ..., z, aa, ab, ...
    Returns a dict {node_id: label} and inverse {label: node_id}.
    """
    labels = {}
    chars  = string.ascii_lowercase
    for i, node in enumerate(node_list):
        if i < 26:
            label = chars[i]
        elif i < 26 + 26 * 26:
            i2 = i - 26
            label = chars[i2 // 26] + chars[i2 % 26]
        else:
            label = str(i)  # fallback for very large graphs
        labels[node] = label
    inverse = {v: k for k, v in labels.items()}
    return labels, inverse


st.set_page_config(page_title="Road Impact AI", layout="wide")
st.title("Road Impact AI")
st.markdown("Road criticality assessment system for urban transport networks.")


@st.cache_resource
def load_artifacts():
    if os.path.exists("data/graph_real.pkl"):
        with open("data/graph_real.pkl", "rb") as f:
            G = pickle.load(f)
        graph_label = "OpenStreetMap"
        od_path = "data/od_matrix_real.npy"
    else:
        from data.generate_graph import load_graph
        G = load_graph()
        graph_label = "Synthetic"
        od_path = "data/od_matrix.npy"

    if os.path.exists("model/random_forest_real.pkl"):
        with open("model/random_forest_real.pkl", "rb") as f:
            model = pickle.load(f)
        fi_path     = "model/feature_importance_real.csv"
        model_label = "real"
    elif os.path.exists("model/random_forest_v2.pkl"):
        with open("model/random_forest_v2.pkl", "rb") as f:
            model = pickle.load(f)
        fi_path     = "model/feature_importance_v2.csv"
        model_label = "v2"
    else:
        with open("model/random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        fi_path     = "model/feature_importance.csv"
        model_label = "v1"

    importance = pd.read_csv(fi_path) if os.path.exists(fi_path) else None

    if os.path.exists(od_path):
        od = np.load(od_path)
    else:
        n = G.number_of_nodes()
        od = np.random.randint(10, 100, size=(n, n))
        np.fill_diagonal(od, 0)

    bridges = set()
    try:
        for u, v in nx.bridges(G):
            bridges.add((u, v))
            bridges.add((v, u))
    except Exception:
        pass

    return model, G, od, importance, model_label, graph_label, bridges


try:
    model, G, od, importance, model_label, graph_label, bridges = load_artifacts()
except Exception as e:
    st.error(f"Failed to load artifacts: {e}")
    st.stop()

node_list    = list(G.nodes())
edges        = list(G.edges())
critical_set = set(tuple(sorted(e)) for e in bridges)
node_label, label_to_node = make_node_labels(node_list)

st.caption(
    f"Graph: {graph_label}  |  Model: {model_label}  |  "
    f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}"
)

# Sidebar
with st.sidebar:
    st.header("Model info")
    st.write(f"**Graph:** {graph_label}")
    st.write(f"**Model:** {model_label}")
    st.write(f"**Nodes:** {G.number_of_nodes()}")
    st.write(f"**Edges:** {G.number_of_edges()}")
    st.write(f"**Bridges:** {len(bridges) // 2}")

    if importance is not None:
        st.divider()
        st.header("Feature Importance")
        fi = importance.sort_values("importance", ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(4, 3))
        ax_fi.barh(fi["feature"], fi["importance"], color="#4472C4")
        ax_fi.set_xlabel("Importance")
        ax_fi.set_title(f"Random Forest ({model_label})")
        plt.tight_layout()
        st.pyplot(fig_fi)

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("Road Network")

    fig, ax = plt.subplots(figsize=(9, 7))

    # Use real geographic coordinates when available
    if all("x" in G.nodes[n] and "y" in G.nodes[n] for n in G.nodes()):
        pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()}
    else:
        pos = nx.spring_layout(G, seed=42, k=1.5)

    normal_e   = [(u, v) for u, v in edges if tuple(sorted((u, v))) not in critical_set]
    critical_e = [(u, v) for u, v in edges if tuple(sorted((u, v))) in critical_set]

    nx.draw_networkx_nodes(G, pos, node_color="#AED6F1", node_size=120, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=normal_e,   edge_color="#808080", width=1.0, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=critical_e, edge_color="#C0392B", width=2.5, ax=ax)

    # Draw alphabetic labels on nodes instead of raw OSM IDs
    nx.draw_networkx_labels(
        G, pos,
        labels=node_label,
        font_size=6,
        font_color="#1a1a1a",
        ax=ax,
    )

    ax.set_title("Red = bridge edges (critical).  Node letters match the dropdown below.")
    ax.axis("off")
    st.pyplot(fig)

    # Build dropdown options using alphabetic labels
    def edge_option(u, v):
        la = node_label[u]
        lb = node_label[v]
        tag = "  [bridge]" if tuple(sorted((u, v))) in critical_set else ""
        return f"{la} --> {lb}{tag}"

    edge_options = [edge_option(u, v) for u, v in edges]

    selected_idx = st.selectbox(
        "Select road for analysis:",
        range(len(edges)),
        format_func=lambda x: edge_options[x],
    )
    sel_u, sel_v = edges[selected_idx]

with col2:
    st.subheader("Analysis")

    la      = node_label[sel_u]
    lb      = node_label[sel_v]
    is_crit = tuple(sorted((sel_u, sel_v))) in critical_set

    st.write(f"Selected road: **{la} --> {lb}**")
    if is_crit:
        st.warning("Bridge edge — only connection between two parts of the network.")

    if st.button("Calculate impact", type="primary"):
        with st.spinner("Running analysis..."):
            data     = G[sel_u][sel_v]
            capacity = data.get("capacity", 20)
            fft      = data.get("weight", 30.0)
            traffic  = np.random.uniform(0.5, 1.5) * TIME_COEFF

            try:
                cent_dict = nx.edge_betweenness_centrality(
                    G, weight="weight", normalized=True
                )
                cent = cent_dict.get(
                    (sel_u, sel_v), cent_dict.get((sel_v, sel_u), 0.0)
                )
            except Exception:
                cent = 0.0

            try:
                alts = len(list(nx.all_simple_paths(G, sel_u, sel_v, cutoff=3)))
                alts = min(max(alts - 1, 0), 10)
            except Exception:
                alts = 0

            features_array = np.array([[
                traffic, capacity, fft, cent, alts,
                1 if is_crit else 0,
                TIME_COEFF,
            ]])

            try:
                impact = float(model.predict(features_array)[0])
            except Exception:
                # fallback for v1 model trained on 5 features
                features_v1 = np.array([[traffic, capacity, fft, cent, alts]])
                impact = float(model.predict(features_v1)[0])

            impact_percent = impact * 100

            c1, c2 = st.columns(2)
            c1.metric("Traffic load",  f"{traffic:.2f}x")
            c2.metric("Centrality",    f"{cent:.4f}")
            c1.metric("Alternatives",  alts)
            c2.metric("Capacity",      capacity)

            st.divider()

            if impact < 0.1:
                st.success(f"Impact: +{impact_percent:.1f}%")
                st.info("Road is NOT critical.")
                st.markdown("Options:\n- Pedestrian zone\n- Bike lane\n- Green space")
            elif impact < 0.3:
                st.warning(f"Impact: +{impact_percent:.1f}%")
                st.info("Road requires analysis before changes.")
                st.markdown("Options:\n- Lane reduction\n- Widened sidewalks")
            else:
                st.error(f"Impact: +{impact_percent:.1f}%")
                st.info("Road is CRITICAL. Changes not recommended.")
                st.markdown(
                    "Options:\n- Keep as-is\n- Improve surface\n- Build bypass first"
                )

            st.divider()
            st.subheader("Explanation")
            if is_crit:
                st.write("- Bridge edge: removing it disconnects the network.")
            if cent > 0.05:
                st.write(
                    f"- High centrality ({cent:.4f}): many routes pass through this road."
                )
            if alts < 3:
                st.write(f"- Few alternative paths ({alts}).")
            if not is_crit and cent <= 0.05 and alts >= 3:
                st.write(
                    "- Low centrality and sufficient alternatives: road is not critical."
                )

st.divider()
st.caption(
    "Data: OpenStreetMap contributors. "
    "Model trained on simulated traffic data."
)
