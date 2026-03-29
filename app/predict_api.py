"""
CLI API для предсказания impact дороги

Использование:
    python app/predict_api.py --list-edges
    python app/predict_api.py --from 0 --to 1
"""

import sys
import os
import argparse
import pickle
import numpy as np

# добавляем путь к корневой папке
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_graph import load_graph
from data.generate_dataset import compute_avg_travel_time_with_demand, calculate_edge_features

# глобальные объекты
_MODEL = None
_GRAPH = None
_OD_MATRIX = None

def load_artifacts():
    """Загружает модель, граф и OD-матрицу"""
    global _MODEL, _GRAPH, _OD_MATRIX
    
    if _MODEL is None:
        with open("model/random_forest.pkl", 'rb') as f:
            _MODEL = pickle.load(f)
        print("✅ Модель загружена")
    
    if _GRAPH is None:
        _GRAPH = load_graph()
        print(f"✅ Граф загружен: {_GRAPH.number_of_nodes()} узлов, {_GRAPH.number_of_edges()} рёбер")
    
    if _OD_MATRIX is None:
        try:
            _OD_MATRIX = np.load("data/od_matrix.npy")
            print(f"✅ OD-матрица загружена: {_OD_MATRIX.shape}")
        except:
            print("⚠️ OD-матрица не найдена")
            _OD_MATRIX = None
    
    return _MODEL, _GRAPH, _OD_MATRIX

def predict_road_impact(u, v):
    """
    Предсказывает impact удаления дороги между узлами u и v
    """
    model, graph, od_matrix = load_artifacts()
    
    # проверяем, существует ли дорога
    if not graph.has_edge(u, v):
        return {
            'error': f'Дорога {u}→{v} не существует',
            'impact': None
        }
    
    # рассчитываем признаки
    features = calculate_edge_features(graph, (u, v), od_matrix)
    
    # текущее среднее время
    if od_matrix is not None:
        current_time = compute_avg_travel_time_with_demand(graph, od_matrix)
    else:
        current_time = 0
    
    # предсказание
    features_array = np.array([[
        features['traffic'],
        features['capacity'],
        features['free_flow_time'],
        features['centrality'],
        features['alternatives']
    ]])
    
    impact = model.predict(features_array)[0]
    impact_percent = impact * 100
    
    # рекомендация
    if impact < 0.1:
        recommendation = "✅ НЕ критична — можно перепрофилировать"
        suggestion = "парк/пешеходная зона/велодорожка"
    elif impact < 0.2:
        recommendation = "⚠️ Осторожно — требует анализа"
        suggestion = "сужение проезжей части/расширение тротуаров"
    else:
        recommendation = "❌ Критична — не рекомендуется изменять"
        suggestion = "оставить как есть"
    
    return {
        'impact': float(impact),
        'impact_percent': round(impact_percent, 1),
        'recommendation': recommendation,
        'suggestion': suggestion,
        'explanation': {
            'traffic': round(features['traffic'], 2),
            'centrality': round(features['centrality'], 3),
            'alternatives': features['alternatives'],
            'free_flow_time': features['free_flow_time']
        },
        'current_avg_time': round(current_time, 2)
    }

def main():
    parser = argparse.ArgumentParser(description='Оценка влияния удаления дороги')
    parser.add_argument('--from', dest='u', type=int, help='Начальный узел')
    parser.add_argument('--to', dest='v', type=int, help='Конечный узел')
    parser.add_argument('--list-edges', action='store_true', help='Показать все дороги')
    
    args = parser.parse_args()
    
    model, graph, _ = load_artifacts()
    
    if args.list_edges:
        print("\n📋 Доступные дороги:")
        for u, v in graph.edges():
            print(f"   {u} → {v}")
        return
    
    if args.u is None or args.v is None:
        print("❌ Укажи дорогу: python app/predict_api.py --from 0 --to 1")
        print("   Или посмотри все: python app/predict_api.py --list-edges")
        return
    
    result = predict_road_impact(args.u, args.v)
    
    if 'error' in result:
        print(f"\n❌ {result['error']}")
        return
    
    print("\n" + "="*50)
    print(f"📊 Анализ дороги {args.u} → {args.v}")
    print("="*50)
    print(f"📈 Impact: +{result['impact_percent']}% времени")
    print(f"💡 Рекомендация: {result['recommendation']}")
    print(f"🎯 Предложение: {result['suggestion']}")
    print("\n🔍 Объяснение:")
    print(f"   • Текущая загрузка: {result['explanation']['traffic']}x")
    print(f"   • Центральность дороги: {result['explanation']['centrality']}")
    print(f"   • Альтернативных путей: {result['explanation']['alternatives']}")
    print(f"   • Свободное время: {result['explanation']['free_flow_time']} мин")
    if result['current_avg_time'] > 0:
        print(f"📊 Среднее время поездки: {result['current_avg_time']} мин")

if __name__ == "__main__":
    main()