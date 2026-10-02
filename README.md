# Road Impact AI (version-2)

Road Impact AI is a decision-support system for urban road infrastructure planning.
It estimates how much removing or closing a specific road would affect average travel
time across a city transport network.

Track: AI inDrive Gov

---

## What is new in version-2

- Interactive OpenStreetMap view in the Streamlit app.
- Road-network overlay on top of the map, with critical bridge edges highlighted.
- Astana map mode in the interface.
- Exact road-segment selection instead of street-level selection.
- Stable segment identifiers based on OSM metadata when available.
- Map bounds limited to the road segments included in the loaded graph.
- Session-stable traffic inputs: random traffic values are generated once per
  Streamlit session and reused on every `Calculate impact` click.
- Updated impact coefficients: bridge status is a weighted factor, not a fixed
  60% score.
- Fallback network layout when geographic coordinates are unavailable.
- Support for real OpenStreetMap artifacts and trained Random Forest models.

---

## Problem

City authorities make decisions about road closures, reconstruction, and repurposing
every year. Without data, these decisions rely on intuition and local knowledge, which
can lead to increased congestion, longer travel times, and overloaded alternative routes.

## Solution

Given a road network and a travel demand matrix, the system simulates the removal of
each road and measures the resulting increase in average travel time. A Random Forest
model is trained on these simulations and learns to predict road criticality from
structural graph features without running a full simulation at inference time.

The result is a score between 0 and 1, along with a recommendation:
repurpose, analyze further, or leave unchanged.

---

## Exact OSM road-segment architecture

A street name is not a reliable road identifier. In OpenStreetMap, one visible
street is usually split into many ways and graph edges. For example, "Sain from
Margulan Street to Abay Avenue" should be stored as a route segment, not as the
whole street named Sain.

Recommended segment identity:

- `segment_id`: stable application ID.
- `osmids`: one or more OSM way IDs that form the selected segment.
- `node_sequence`: ordered OSM node IDs from the start intersection to the end
  intersection.
- `from_intersection`: OSM node nearest to the Sain / Margulan intersection.
- `to_intersection`: OSM node nearest to the Sain / Abay intersection.
- `geometry`: ordered polyline used by the map.
- `features`: capacity, travel time, centrality, alternatives, traffic, and
  bridge/connectivity flags used by the model.

Selection flow for a real corridor:

1. Find candidate OSM ways whose `name` matches the main street, for example
   Sain.
2. Find the nearest graph nodes to the two boundary intersections, for example
   Margulan and Abay.
3. Run shortest path between those boundary nodes while preferring ways with the
   selected street name.
4. Save the resulting ordered edge list as one logical road segment.
5. Use that segment ID in the UI and calculations instead of selecting by street
   name alone.

The current app selects individual graph segments from the loaded road graph.
After regenerating `graph_real.pkl`, the dropdown will also show preserved OSM
metadata such as street name, OSM way ID, and length.

---

## Setup

### Requirements

- Python 3.9 or higher

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the web interface

```bash
streamlit run app/streamlit_app.py
```

The app automatically loads the best available artifacts:

1. `data/graph_real.pkl` and `model/random_forest_real.pkl`, if present.
2. `model/random_forest_v2.pkl`, if the real model is not present.
3. The original synthetic graph/model as a fallback.

In the interface:

1. Select a road from the dropdown.
2. Review it on the OpenStreetMap view.
3. Click `Calculate impact`.
4. Read the impact score and recommendation.

Traffic values are generated once per Streamlit session. Repeated clicks on
`Calculate impact` reuse the same session data, so the result does not change just
because the button was pressed again. A new browser session or app restart creates a
fresh traffic profile.

---

## Regenerate real OpenStreetMap data

The real graph generator uses `data/generate_graph_real.py`.

For Astana, set the city in that file:

```python
CITY = "Astana, Kazakhstan"
```

Then run the pipeline:

### Step 1 - Download the road graph

```bash
python data/generate_graph_real.py
```

Creates:

- `data/graph_real.pkl`

This step requires an internet connection. It can take 1-3 minutes depending on
connection speed and the selected graph size.

### Step 2 - Generate the OD matrix

```bash
python data/od_matrix.py --real
```

Creates:

- `data/od_matrix_real.npy`

### Step 3 - Generate the training dataset

```bash
python data/generate_dataset_real.py
```

Creates:

- `data/dataset_real.csv`

This step can take 5-15 minutes depending on the graph size.

### Step 4 - Train the real-data model

```bash
python model/train_model_real.py
```

Creates:

- `model/random_forest_real.pkl`
- `model/feature_importance_real.csv`

### Step 5 - Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## Optional v2 synthetic training pipeline

The v2 synthetic pipeline includes additional features such as traffic level,
critical-edge flag, and time coefficient.

```bash
python data/generate_dataset_v2.py
python model/train_model_v2.py
```

Creates:

- `data/dataset_v2.csv`
- `model/random_forest_v2.pkl`
- `model/feature_importance_v2.csv`

---

## CLI alternative

List all available roads:

```bash
python app/predict_api.py --list-edges
```

Analyze a specific road by node IDs:

```bash
python app/predict_api.py --from 0 --to 1
```

---

## Visualize the graph only

```bash
python app/visualize_graph.py
```

Creates:

- `docs/road_graph.png`

---

## Notes

- OpenStreetMap tiles are provided by OpenStreetMap contributors.
- The model is trained on simulated traffic data.
- The Streamlit map overlays the loaded graph when nodes contain geographic
  coordinates. If not, the app shows a network layout instead.
