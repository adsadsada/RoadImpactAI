import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import pickle
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURE_COLS = [
    "traffic", "capacity", "free_flow_time",
    "centrality", "alternatives", "is_critical", "time_coeff",
]

# Fixed time coefficient — peak hour average.
# Time-of-day selection was removed: a road that is critical at peak hour
# cannot be repurposed just because it is quiet at night.
TIME_COEFF = 1.3
ASTANA_CENTER = [51.1694, 71.4491]


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


def first_edge_data(graph, u, v):
    data = graph.get_edge_data(u, v, default={})
    if not isinstance(data, dict):
        return {}
    if any(key in data for key in ("geometry", "weight", "capacity", "length")):
        return data
    for value in data.values():
        if isinstance(value, dict):
            return value
    return data


def edge_latlon_path(graph, u, v):
    if not all("x" in graph.nodes[n] and "y" in graph.nodes[n] for n in (u, v)):
        return None

    data = first_edge_data(graph, u, v)
    geometry = data.get("geometry")
    if geometry is not None and hasattr(geometry, "coords"):
        return [[lat, lon] for lon, lat in geometry.coords]

    return [
        [graph.nodes[u]["y"], graph.nodes[u]["x"]],
        [graph.nodes[v]["y"], graph.nodes[v]["x"]],
    ]


def graph_has_geo_coordinates(graph):
    return all("x" in graph.nodes[n] and "y" in graph.nodes[n] for n in graph.nodes())


def graph_session_signature(graph, graph_edges):
    sample_edges = graph_edges[:10] + graph_edges[-10:]
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        tuple((str(u), str(v)) for u, v in sample_edges),
    )


def get_session_input_data(graph, graph_edges):
    signature = graph_session_signature(graph, graph_edges)
    if st.session_state.get("impact_data_signature") != signature:
        st.session_state["impact_data_signature"] = signature
        st.session_state["impact_data"] = {
            "traffic_by_edge": {
                (u, v): float(np.random.uniform(0.5, 1.5) * TIME_COEFF)
                for u, v in graph_edges
            }
        }
    return st.session_state["impact_data"]


def render_astana_osm_map(graph, graph_edges, labels, critical_edges, selected_edge):
    map_edges = []
    bounds_points = []

    for u, v in graph_edges:
        coords = edge_latlon_path(graph, u, v)
        if not coords:
            continue
        is_critical = tuple(sorted((u, v))) in critical_edges
        map_edges.append({
            "coords": coords,
            "critical": is_critical,
            "selected": (u, v) == selected_edge,
            "label": f"{labels[u]} -> {labels[v]}" + (" [critical]" if is_critical else ""),
        })
        bounds_points.extend(coords)

    if bounds_points:
        center = [
            sum(point[0] for point in bounds_points) / len(bounds_points),
            sum(point[1] for point in bounds_points) / len(bounds_points),
        ]
        zoom = 12
    else:
        center = ASTANA_CENTER
        zoom = 11

    payload = json.dumps({
        "center": center,
        "zoom": zoom,
        "edges": map_edges,
    })

    html = """
    <link
      rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
      integrity="sha256-p4NxAoJBhIINfQPDmJOSiGakgVbM9h9G17Cf9Iibn4A="
      crossorigin=""
    />
    <script
      src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
      integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
      crossorigin=""
    ></script>
    <div id="astana-map" style="height: 620px; width: 100%; border-radius: 8px; overflow: hidden;"></div>
    <script>
      const data = __MAP_DATA__;
      const map = L.map('astana-map', { preferCanvas: true }).setView(data.center, data.zoom);

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap contributors'
      }).addTo(map);

      const layerGroup = L.featureGroup().addTo(map);
      let selectedLayer = null;

      data.edges.forEach((edge) => {
        const style = {
          color: edge.selected ? '#f1c40f' : (edge.critical ? '#c0392b' : '#2f80ed'),
          weight: edge.selected ? 7 : (edge.critical ? 4 : 2),
          opacity: edge.selected ? 1 : (edge.critical ? 0.9 : 0.55)
        };
        const line = L.polyline(edge.coords, style).bindTooltip(edge.label);
        line.addTo(layerGroup);
        if (edge.selected) {
          selectedLayer = line;
        }
      });

      if (selectedLayer) {
        map.fitBounds(selectedLayer.getBounds(), { padding: [80, 80], maxZoom: 16 });
      } else if (layerGroup.getLayers().length > 0) {
        map.fitBounds(layerGroup.getBounds(), { padding: [30, 30], maxZoom: 13 });
      }
    </script>
    """.replace("__MAP_DATA__", payload)

    components.html(html, height=640)


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
session_input_data = get_session_input_data(G, edges)

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
    st.subheader("Astana OpenStreetMap")

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

    if graph_has_geo_coordinates(G):
        render_astana_osm_map(G, edges, node_label, critical_set, (sel_u, sel_v))
    else:
        st.info("OpenStreetMap coordinates are unavailable, showing network layout instead.")
        fig, ax = plt.subplots(figsize=(9, 7))
        pos = nx.spring_layout(G, seed=42, k=1.5)

        normal_e   = [(u, v) for u, v in edges if tuple(sorted((u, v))) not in critical_set]
        critical_e = [(u, v) for u, v in edges if tuple(sorted((u, v))) in critical_set]

        nx.draw_networkx_nodes(G, pos, node_color="#AED6F1", node_size=120, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=normal_e,   edge_color="#808080", width=1.0, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=critical_e, edge_color="#C0392B", width=2.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=[(sel_u, sel_v)], edge_color="#F1C40F", width=4.0, ax=ax)

        nx.draw_networkx_labels(
            G, pos,
            labels=node_label,
            font_size=6,
            font_color="#1a1a1a",
            ax=ax,
        )

        ax.set_title("Red = bridge edges (critical). Yellow = selected road.")
        ax.axis("off")
        st.pyplot(fig)

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
            data     = first_edge_data(G, sel_u, sel_v)
            capacity = data.get("capacity", 20)
            fft      = data.get("weight", 30.0)
            traffic  = session_input_data["traffic_by_edge"][(sel_u, sel_v)]

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
