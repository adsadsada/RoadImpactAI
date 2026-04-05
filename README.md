# Road Impact AI

Road Impact AI is a decision-support system for urban road infrastructure planning.
It estimates how much removing or closing a specific road would affect average travel
time across the entire city network.

Track: AI inDrive Gov

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

The result is a score between 0 and 1 (the impact), along with a recommendation:
repurpose, analyze further, or leave unchanged.

---

## Setup and Installation

### Requirements

- Python 3.9 or higher

### Install dependencies

```
pip install -r requirements.txt
```

---

## Running the project

Steps must be run in order. Each script produces files that the next one depends on.

### Step 1 — Download the road graph

Downloads the Almaty road network from OpenStreetMap and saves a subgraph of 500 nodes.

```
python data/generate_graph_real.py
```

Creates `data/graph_real.pkl` file

This step requires an internet connection. It takes 1-3 minutes depending on connection speed.

---

### Step 2 — Generate the OD matrix

```
python data/od_matrix.py --real
```

Creates `data/od_matrix_real.npy` file

---

### Step 3 — Generate the training dataset

```
python data/generate_dataset_real.py
```

Creates `data/dataset_real.csv` file

This step takes approximately 5-15 minutes depending on the size of the graph.

---

### Step 4 — Train the model

```
python model/train_model_real.py
```

Creates `model/random_forest_real.pkl`, `model/feature_importance_real.csv` files

---

### Step 5 — Run the web interface

```
streamlit run app/streamlit_app.py
```

Select a road from the dropdown, click "Calculate impact", and read the result.
Each node on the map is labeled with a letter (a, b, c, ...). The dropdown uses
the same letters, for example: `a --> b` or `f --> k [bridge]`.

---

### CLI alternative

List all available roads:

```
python app/predict_api.py --list-edges
```

Analyze a specific road by node IDs:

```
python app/predict_api.py --from 0 --to 1
```

---

### Visualize the graph only

```
python app/visualize_graph.py
```

Creates `docs/road_graph.png` file

---
