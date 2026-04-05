import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATASET_PATH    = "data/dataset_real.csv"
MODEL_PATH      = "model/random_forest_real.pkl"
IMPORTANCE_PATH = "model/feature_importance_real.csv"

FEATURE_COLS = [
    "traffic",
    "capacity",
    "free_flow_time",
    "centrality",
    "alternatives",
    "is_critical",
    "time_coeff",
]

RF_PARAMS = dict(
    n_estimators      = 150,
    max_depth         = 12,
    min_samples_split = 4,
    min_samples_leaf  = 2,
    random_state      = 42,
    n_jobs            = -1,
)


def train(dataset_path=DATASET_PATH):
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Run first: python data/generate_dataset_real.py")
        return None, None

    df = pd.read_csv(dataset_path)
    print(f"Dataset: {len(df)} rows, {len(FEATURE_COLS)} features")
    print(f"Impact: min={df['impact'].min():.4f}  mean={df['impact'].mean():.4f}  max={df['impact'].max():.4f}")

    X = df[FEATURE_COLS]
    y = df["impact"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.4f}")
    print(f"R2:  {r2:.4f}")

    importance = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nFeature importance:")
    for _, row in importance.iterrows():
        bar = "#" * int(row["importance"] * 50)
        print(f"  {row['feature']:<18} {row['importance']:.3f}  {bar}")

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    importance.to_csv(IMPORTANCE_PATH, index=False)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Feature importance saved to {IMPORTANCE_PATH}")
    print("Next step: streamlit run app/streamlit_app.py")

    return model, importance


if __name__ == "__main__":
    train()
