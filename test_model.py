import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('data/dataset.csv')
X = df[['traffic', 'capacity', 'free_flow_time', 'centrality', 'alternatives']]
y = df['impact']

with open('model/random_forest.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X)
print(f'MAE: {mean_absolute_error(y, y_pred):.4f}')
print(f'R²: {r2_score(y, y_pred):.4f}')