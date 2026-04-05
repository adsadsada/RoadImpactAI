import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

os.makedirs("model", exist_ok=True)

FEATURE_COLS_V2 = [
    'traffic', 'capacity', 'free_flow_time',
    'centrality', 'alternatives', 'is_critical', 'time_coeff'
]

def train_model_v2(dataset_path="data/dataset_v2.csv"):
    print("📊 Загрузка датасета v2...")
    df = pd.read_csv(dataset_path)

    X = df[FEATURE_COLS_V2]
    y = df['impact']

    print(f"Размер выборки: {len(df)}")
    print(f"Признаков: {len(FEATURE_COLS_V2)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n🔄 Обучение Random Forest v2...")
    print("   n_estimators=150, max_depth=12")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\n✅ Модель v2 обучена")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²:  {r2:.4f}")

    importance = pd.DataFrame({
        'feature':    FEATURE_COLS_V2,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📊 Важность признаков:")
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 40)
        print(f"   {row['feature']:<18} {row['importance']:.3f}  {bar}")

    with open("model/random_forest_v2.pkl", 'wb') as f:
        pickle.dump(model, f)

    importance.to_csv("model/feature_importance_v2.csv", index=False)

    print("\n💾 Сохранено:")
    print("   - model/random_forest_v2.pkl")
    print("   - model/feature_importance_v2.csv")

    return model, importance, {'mae': mae, 'r2': r2}


if __name__ == "__main__":
    train_model_v2()
