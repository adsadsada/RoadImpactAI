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

# Fixed time coefficient: peak-hour average.
# Time-of-day selection was removed because a structurally critical road should
# not become "safe to remove" just because the current hour is quiet.
TIME_COEFF = 1.3
ASTANA_CENTER = [51.1694, 71.4491]

MAP_TILE_URL = "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
MAP_TILE_ATTRIBUTION = "&copy; OpenStreetMap contributors &copy; CARTO"
BLANK_TILE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

# Blended impact coefficients used after the ML estimate. A bridge is now a
# factor in the final score, not a hard-coded 60% floor.
IMPACT_WEIGHTS = {
    "model": 0.35,
    "traffic": 0.18,
    "centrality": 0.20,
    "alternatives": 0.12,
    "capacity": 0.07,
    "free_flow_time": 0.04,
    "bridge": 0.04,
}


def make_node_labels(node_list):
    """
    Assign short alphabetic labels to nodes: a, b, ..., z, aa, ab, ...
    Returns a dict {node_id: label} and inverse {label: node_id}.
    """
    labels = {}
    chars = string.ascii_lowercase
    for i, node in enumerate(node_list):
        if i < 26:
            label = chars[i]
        elif i < 26 + 26 * 26:
            i2 = i - 26
            label = chars[i2 // 26] + chars[i2 % 26]
        else:
            label = str(i)
        labels[node] = label
    inverse = {v: k for k, v in labels.items()}
    return labels, inverse


def first_edge_data(graph, u, v):
    data = graph.get_edge_data(u, v, default={})
    if not isinstance(data, dict):
        return {}
    if any(key in data for key in (
        "geometry", "weight", "capacity", "length", "name", "osmid"
    )):
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
                (u, v): float(np.random.uniform(0.45, 1.65) * TIME_COEFF)
                for u, v in graph_edges
            }
        }
    return st.session_state["impact_data"]


def clean_osm_value(value):
    if isinstance(value, (list, tuple, set)):
        items = [str(item) for item in value if item is not None]
        return ", ".join(items[:3])
    if value is None:
        return ""
    return str(value)


def format_length_meters(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    return f"{number:.0f} m"


def normalize(value, low, high):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if high <= low:
        return 0.0
    return float(np.clip((number - low) / (high - low), 0.0, 1.0))


def clamp01(value):
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def make_segment_id(index, u, v, data):
    osmid = clean_osm_value(data.get("osmid"))
    if osmid:
        return f"osm:{osmid}:{u}:{v}"
    return f"edge:{index}:{u}:{v}"


def build_road_segments(graph, graph_edges, labels, critical_edges):
    segments = []

    for index, (u, v) in enumerate(graph_edges):
        data = first_edge_data(graph, u, v)
        node_label = f"{labels[u]} -> {labels[v]}"
        street_name = clean_osm_value(data.get("name"))
        osmid = clean_osm_value(data.get("osmid"))
        length = format_length_meters(data.get("length"))
        is_critical = tuple(sorted((u, v))) in critical_edges

        title = street_name or f"Road segment {index + 1:03d}"
        details = [node_label]
        if length:
            details.append(length)
        if osmid:
            details.append(f"OSM {osmid}")
        if is_critical:
            details.append("bridge")

        display_label = f"{index + 1:03d} | {title} | " + " | ".join(details)
        map_label = display_label.replace(" | ", "<br>")

        segments.append({
            "index": index,
            "edge": (u, v),
            "segment_id": make_segment_id(index, u, v, data),
            "title": title,
            "display_label": display_label,
            "map_label": map_label,
            "coords": edge_latlon_path(graph, u, v),
            "critical": is_critical,
            "data": data,
        })

    return segments


def calculate_adjusted_impact(model_impact, traffic, capacity, fft, cent, alts, is_crit):
    scores = {
        "model": clamp01(model_impact),
        "traffic": normalize(traffic, 0.40, 2.30),
        "centrality": normalize(cent, 0.00, 0.12),
        "alternatives": 1.0 - normalize(alts, 0.0, 10.0),
        "capacity": normalize(capacity, 20.0, 140.0),
        "free_flow_time": normalize(fft, 5.0, 120.0),
        "bridge": 1.0 if is_crit else 0.0,
    }
    weighted = sum(IMPACT_WEIGHTS[name] * scores[name] for name in IMPACT_WEIGHTS)
    return clamp01(weighted), scores


def render_astana_osm_map(road_segments, selected_index):
    map_edges = []
    bounds_points = []

    for segment in road_segments:
        coords = segment["coords"]
        if not coords:
            continue
        selected = segment["index"] == selected_index
        map_edges.append({
            "index": segment["index"],
            "segment_id": segment["segment_id"],
            "coords": coords,
            "critical": segment["critical"],
            "selected": selected,
            "label": segment["map_label"],
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
        "tileUrl": MAP_TILE_URL,
        "tileAttribution": MAP_TILE_ATTRIBUTION,
        "blankTileUrl": BLANK_TILE_URL,
    })

    html = """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
      #astana-map {
        height: 620px;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        background: #eef2f6;
      }
      .leaflet-container {
        background: #eef2f6;
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .segment-tooltip {
        font-size: 12px;
        line-height: 1.35;
      }
    </style>
    <div id="astana-map"></div>
    <script>
      const data = __MAP_DATA__;
      const map = L.map('astana-map', {
        preferCanvas: true,
        zoomControl: true,
        worldCopyJump: false
      }).setView(data.center, data.zoom);

      L.tileLayer(data.tileUrl, {
        subdomains: 'abcd',
        maxZoom: 20,
        detectRetina: true,
        updateWhenIdle: true,
        updateWhenZooming: false,
        keepBuffer: 4,
        crossOrigin: true,
        errorTileUrl: data.blankTileUrl,
        attribution: data.tileAttribution
      }).addTo(map);

      const layerGroup = L.featureGroup().addTo(map);
      const allBounds = L.latLngBounds([]);
      const selectedBounds = L.latLngBounds([]);
      let selectedLayer = null;
      let selectedEdge = null;

      data.edges.forEach((edge) => {
        const style = {
          color: edge.selected ? '#f1c40f' : (edge.critical ? '#c0392b' : '#2f80ed'),
          weight: edge.selected ? 8 : (edge.critical ? 4 : 2.5),
          opacity: edge.selected ? 1 : (edge.critical ? 0.9 : 0.58),
          lineCap: 'round',
          lineJoin: 'round'
        };

        const line = L.polyline(edge.coords, style).bindTooltip(edge.label, {
          sticky: true,
          className: 'segment-tooltip'
        });
        line.addTo(layerGroup);
        edge.coords.forEach((point) => allBounds.extend(point));

        if (edge.selected) {
          selectedLayer = line;
          selectedEdge = edge;
          edge.coords.forEach((point) => selectedBounds.extend(point));
        }
      });

      if (allBounds.isValid()) {
        const limitedBounds = allBounds.pad(0.35);
        map.setMaxBounds(limitedBounds);
        map.on('drag', () => map.panInsideBounds(limitedBounds, { animate: false }));
      }

      if (selectedLayer && selectedBounds.isValid()) {
        selectedLayer.bringToFront();
        map.fitBounds(selectedBounds, { padding: [90, 90], maxZoom: 17 });

        const start = selectedEdge.coords[0];
        const end = selectedEdge.coords[selectedEdge.coords.length - 1];
        L.circleMarker(start, {
          radius: 5,
          color: '#14532d',
          fillColor: '#22c55e',
          fillOpacity: 1,
          weight: 2
        }).addTo(layerGroup).bindTooltip('segment start');
        L.circleMarker(end, {
          radius: 5,
          color: '#7f1d1d',
          fillColor: '#ef4444',
          fillOpacity: 1,
          weight: 2
        }).addTo(layerGroup).bindTooltip('segment end');
      } else if (allBounds.isValid()) {
        map.fitBounds(allBounds, { padding: [40, 40], maxZoom: 14 });
      }

      L.control.scale({ metric: true, imperial: false }).addTo(map);
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
        fi_path = "model/feature_importance_real.csv"
        model_label = "real"
    elif os.path.exists("model/random_forest_v2.pkl"):
        with open("model/random_forest_v2.pkl", "rb") as f:
            model = pickle.load(f)
        fi_path = "model/feature_importance_v2.csv"
        model_label = "v2"
    else:
        with open("model/random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        fi_path = "model/feature_importance.csv"
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

node_list = list(G.nodes())
edges = list(G.edges())

if not edges:
    st.error("The loaded graph does not contain road segments.")
    st.stop()

critical_set = set(tuple(sorted(e)) for e in bridges)
node_label, label_to_node = make_node_labels(node_list)
road_segments = build_road_segments(G, edges, node_label, critical_set)
session_input_data = get_session_input_data(G, edges)

st.caption(
    f"Graph: {graph_label}  |  Model: {model_label}  |  "
    f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}"
)

with st.sidebar:
    st.header("Model info")
    st.write(f"**Graph:** {graph_label}")
    st.write(f"**Model:** {model_label}")
    st.write(f"**Nodes:** {G.number_of_nodes()}")
    st.write(f"**Edges:** {G.number_of_edges()}")
    st.write(f"**Bridges:** {len(bridges) // 2}")

    st.divider()
    st.header("Impact weights")
    for factor, weight in IMPACT_WEIGHTS.items():
        st.write(f"**{factor}:** {weight:.0%}")

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

    selected_idx = st.selectbox(
        "Select road segment for analysis:",
        range(len(road_segments)),
        format_func=lambda x: road_segments[x]["display_label"],
    )
    selected_segment = road_segments[selected_idx]
    sel_u, sel_v = selected_segment["edge"]

    if graph_has_geo_coordinates(G):
        render_astana_osm_map(road_segments, selected_idx)
    else:
        st.info("OpenStreetMap coordinates are unavailable, showing network layout instead.")
        fig, ax = plt.subplots(figsize=(9, 7))
        pos = nx.spring_layout(G, seed=42, k=1.5)

        normal_e = [(u, v) for u, v in edges if tuple(sorted((u, v))) not in critical_set]
        critical_e = [(u, v) for u, v in edges if tuple(sorted((u, v))) in critical_set]

        nx.draw_networkx_nodes(G, pos, node_color="#AED6F1", node_size=120, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=normal_e, edge_color="#808080", width=1.0, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=critical_e, edge_color="#C0392B", width=2.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=[(sel_u, sel_v)], edge_color="#F1C40F", width=4.0, ax=ax)

        nx.draw_networkx_labels(
            G, pos,
            labels=node_label,
            font_size=6,
            font_color="#1a1a1a",
            ax=ax,
        )

        ax.set_title("Red = bridge edges. Yellow = selected road segment.")
        ax.axis("off")
        st.pyplot(fig)

with col2:
    st.subheader("Analysis")

    la = node_label[sel_u]
    lb = node_label[sel_v]
    is_crit = selected_segment["critical"]

    st.write(f"Selected segment: **{selected_segment['title']}**")
    st.caption(f"{la} -> {lb} | {selected_segment['segment_id']}")
    if is_crit:
        st.warning("Bridge edge: this is an important connectivity factor, not a fixed 60% score.")

    if st.button("Calculate impact", type="primary"):
        with st.spinner("Running analysis..."):
            data = first_edge_data(G, sel_u, sel_v)
            capacity = data.get("capacity", 20)
            fft = data.get("weight", 30.0)
            traffic = session_input_data["traffic_by_edge"][(sel_u, sel_v)]

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

            feature_values = {
                "traffic": traffic,
                "capacity": capacity,
                "free_flow_time": fft,
                "centrality": cent,
                "alternatives": alts,
                "is_critical": 1 if is_crit else 0,
                "time_coeff": TIME_COEFF,
            }
            feature_frame = pd.DataFrame([feature_values], columns=FEATURE_COLS)

            try:
                model_impact = float(model.predict(feature_frame)[0])
            except Exception:
                features_v1 = pd.DataFrame([{
                    "traffic": traffic,
                    "capacity": capacity,
                    "free_flow_time": fft,
                    "centrality": cent,
                    "alternatives": alts,
                }])
                model_impact = float(model.predict(features_v1)[0])

            impact, factor_scores = calculate_adjusted_impact(
                model_impact, traffic, capacity, fft, cent, alts, is_crit
            )
            impact_percent = impact * 100

            c1, c2 = st.columns(2)
            c1.metric("Traffic load", f"{traffic:.2f}x")
            c2.metric("Centrality", f"{cent:.4f}")
            c1.metric("Alternatives", alts)
            c2.metric("Capacity", capacity)

            st.divider()

            if impact < 0.18:
                st.success(f"Impact: +{impact_percent:.1f}%")
                st.info("Road segment is not critical.")
                st.markdown("Options:\n- Pedestrian zone\n- Bike lane\n- Green space")
            elif impact < 0.38:
                st.warning(f"Impact: +{impact_percent:.1f}%")
                st.info("Road segment requires additional analysis before changes.")
                st.markdown("Options:\n- Lane reduction\n- Widened sidewalks")
            else:
                st.error(f"Impact: +{impact_percent:.1f}%")
                st.info("Road segment is critical. Changes are not recommended yet.")
                st.markdown(
                    "Options:\n- Keep as-is\n- Improve surface\n- Build bypass first"
                )

            st.divider()
            st.subheader("Explanation")
            st.write(f"- Model estimate: {clamp01(model_impact) * 100:.1f}%.")
            st.write(
                "- Final score is blended from model output, traffic, centrality, "
                "alternatives, capacity, free-flow time, and bridge status."
            )
            if is_crit:
                st.write("- Bridge status adds risk, but it no longer locks impact at 60%.")
            if cent > 0.05:
                st.write(f"- High centrality ({cent:.4f}): many routes pass through this segment.")
            if alts < 3:
                st.write(f"- Few alternative paths ({alts}).")
            if not is_crit and cent <= 0.05 and alts >= 3:
                st.write("- Low centrality and sufficient alternatives reduce the score.")

            with st.expander("Factor scores"):
                for factor, score in factor_scores.items():
                    st.write(f"**{factor}:** {score:.2f}")

st.divider()
st.caption(
    "Data: OpenStreetMap contributors. "
    "Model trained on simulated traffic data."
)
