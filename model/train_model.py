import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

os.makedirs("model", exist_ok=True)

def train_model(dataset_path="data/dataset.csv"):
    print("📊 Загрузка датасета...")
    df = pd.read_csv(dataset_path)
    
    feature_cols = ['traffic', 'capacity', 'free_flow_time', 'centrality', 'alternatives']
    X = df[feature_cols]
    y = df['impact']
    
    print(f"Размер выборки: {len(df)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("🔄 Обучение Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n✅ Модель обучена")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 Важность признаков:")
    for _, row in importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    with open("model/random_forest.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    importance.to_csv("model/feature_importance.csv", index=False)
    
    print("\n💾 Модель сохранена:")
    print("   - model/random_forest.pkl")
    print("   - model/feature_importance.csv")
    
    return model, importance

if __name__ == "__main__":
    train_model()