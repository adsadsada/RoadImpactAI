import pandas as pd
import pickle
import os
from sklearn.metrics import mean_absolute_error, r2_score

# Load best available dataset and model
if os.path.exists("data/dataset_real.csv") and os.path.exists("model/random_forest_real.pkl"):
    dataset = "data/dataset_real.csv"
    model_path = "model/random_forest_real.pkl"
    features = ["traffic", "capacity", "free_flow_time", "centrality", "alternatives", "is_critical", "time_coeff"]
elif os.path.exists("data/dataset_v2.csv") and os.path.exists("model/random_forest_v2.pkl"):
    dataset = "data/dataset_v2.csv"
    model_path = "model/random_forest_v2.pkl"
    features = ["traffic", "capacity", "free_flow_time", "centrality", "alternatives", "is_critical", "time_coeff"]
else:
    dataset = "data/dataset.csv"
    model_path = "model/random_forest.pkl"
    features = ["traffic", "capacity", "free_flow_time", "centrality", "alternatives"]

df = pd.read_csv(dataset)
X  = df[features]
y  = df["impact"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

y_pred = model.predict(X)
print(f"Dataset:  {dataset}")
print(f"Model:    {model_path}")
print(f"Rows:     {len(df)}")
print(f"MAE:      {mean_absolute_error(y, y_pred):.4f}")
print(f"R2:       {r2_score(y, y_pred):.4f}")
