import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import os

FEATURE_COLS_V1   = ["traffic", "capacity", "free_flow_time", "centrality", "alternatives"]
FEATURE_COLS_V2   = ["traffic", "capacity", "free_flow_time", "centrality", "alternatives",
                     "is_critical", "time_coeff"]
FEATURE_COLS_REAL = FEATURE_COLS_V2  # same columns, different training data


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "mae": mean_absolute_error(y, pred),
        "r2":  r2_score(y, pred),
    }


def compare():
    versions = []

    if os.path.exists("model/random_forest.pkl") and os.path.exists("data/dataset.csv"):
        df = pd.read_csv("data/dataset.csv")
        X  = df[FEATURE_COLS_V1]
        y  = df["impact"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        m = load_model("model/random_forest.pkl")
        s = evaluate(m, X_test, y_test)
        fi = pd.read_csv("model/feature_importance.csv") if os.path.exists("model/feature_importance.csv") else None
        versions.append(("v1", len(df), len(FEATURE_COLS_V1), 100, 10, s, fi))

    if os.path.exists("model/random_forest_v2.pkl") and os.path.exists("data/dataset_v2.csv"):
        df = pd.read_csv("data/dataset_v2.csv")
        X  = df[FEATURE_COLS_V2]
        y  = df["impact"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        m = load_model("model/random_forest_v2.pkl")
        s = evaluate(m, X_test, y_test)
        fi = pd.read_csv("model/feature_importance_v2.csv") if os.path.exists("model/feature_importance_v2.csv") else None
        versions.append(("v2", len(df), len(FEATURE_COLS_V2), 150, 12, s, fi))

    if os.path.exists("model/random_forest_real.pkl") and os.path.exists("data/dataset_real.csv"):
        df = pd.read_csv("data/dataset_real.csv")
        X  = df[FEATURE_COLS_REAL]
        y  = df["impact"]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        m = load_model("model/random_forest_real.pkl")
        s = evaluate(m, X_test, y_test)
        fi = pd.read_csv("model/feature_importance_real.csv") if os.path.exists("model/feature_importance_real.csv") else None
        versions.append(("real", len(df), len(FEATURE_COLS_REAL), 150, 12, s, fi))

    if not versions:
        print("No models found.")
        return

    header = f"{'Metric':<22}" + "".join(f"{v[0]:>12}" for v in versions)
    print(header)
    print("-" * len(header))
    print(f"{'Dataset rows':<22}" + "".join(f"{v[1]:>12}" for v in versions))
    print(f"{'Features':<22}"     + "".join(f"{v[2]:>12}" for v in versions))
    print(f"{'n_estimators':<22}" + "".join(f"{v[3]:>12}" for v in versions))
    print(f"{'max_depth':<22}"    + "".join(f"{v[4]:>12}" for v in versions))
    print(f"{'MAE':<22}"          + "".join(f"{v[5]['mae']:>12.4f}" for v in versions))
    print(f"{'R2':<22}"           + "".join(f"{v[5]['r2']:>12.4f}" for v in versions))

    # Feature importance table
    all_features = list(dict.fromkeys(
        FEATURE_COLS_V1 + FEATURE_COLS_V2
    ))
    print(f"\n{'Feature':<20}" + "".join(f"{v[0]:>12}" for v in versions))
    print("-" * (20 + 12 * len(versions)))
    for feat in all_features:
        row = f"{feat:<20}"
        for v in versions:
            fi = v[6]
            if fi is not None:
                val = fi[fi["feature"] == feat]["importance"]
                row += f"{float(val.iloc[0]) if len(val) > 0 else 0.0:>12.3f}"
            else:
                row += f"{'N/A':>12}"
        print(row)


if __name__ == "__main__":
    compare()
