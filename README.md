# 🛣️ Road Impact AI

> AI-система оценки критичности дорог городской транспортной сети.
> Трек: **AI inDrive Gov** | Этап 2

---

## 🧠 Проблема

Городские власти ежегодно принимают решения о перекрытии, реконструкции или перепрофилировании дорог. Без аналитики эти решения принимаются интуитивно — и могут привести к пробкам, росту времени в дороге и перегрузке альтернативных маршрутов.

## 💡 Решение

**Road Impact AI** за секунды отвечает на вопрос:
> *«Если убрать эту дорогу — насколько ухудшится трафик по всему городу?»*

Система использует граф дорог, матрицу спроса и ML-модель (Random Forest), обученную на симуляции тысяч сценариев.

---

## 🏗️ Архитектура

```
Граф дорог (15 узлов)
        │
        ▼
BPR-симуляция трафика  ──►  dataset_v2.csv (315 записей)
        │
        ▼
Random Forest v2 (7 признаков)  ──►  random_forest_v2.pkl
        │
        ▼
Streamlit UI  +  CLI API
```

Подробнее: [`docs/architecture.md`](docs/architecture.md)

---

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Генерация данных и обучение (если нужно воспроизвести)
```bash
python data/generate_graph.py
python data/od_matrix.py
python data/generate_dataset_v2.py
python model/train_model_v2.py
```

### 3. Запуск веб-интерфейса
```bash
streamlit run app/streamlit_app.py
```

### 4. CLI API
```bash
# Список всех дорог
python app/predict_api.py --list-edges

# Анализ конкретной дороги (узел 1 → узел 5)
python app/predict_api.py --from 1 --to 5
```

### 5. Сравнение моделей v1 и v2
```bash
python compare_models.py
```

---

## 📊 Результаты модели

| Параметр         | v1        | v2        |
|-----------------|-----------|-----------|
| Датасет         | 210 записей | 315 записей |
| Признаков       | 5         | 7         |
| n_estimators    | 100       | 150       |
| max_depth       | 10        | 12        |
| MAE             | 0.032     | 0.042     |
| R²              | 0.966     | 0.831     |

Новые признаки v2: `is_critical` (мост?) и `time_coeff` (время суток).

---

## 📁 Структура проекта

```
RoadImpactAI/
├── data/
│   ├── generate_graph.py        # Граф города (15 узлов, районы + мосты)
│   ├── od_matrix.py             # OD-матрица спроса
│   ├── generate_dataset.py      # Симуляция v1
│   ├── generate_dataset_v2.py   # Симуляция v2 (критические рёбра, 5 периодов)
│   ├── dataset.csv              # Датасет v1 (210 записей)
│   ├── dataset_v2.csv           # Датасет v2 (315 записей)
│   ├── graph.pkl                # Сохранённый граф
│   └── od_matrix.npy            # Сохранённая OD-матрица
├── model/
│   ├── train_model.py           # Обучение v1
│   ├── train_model_v2.py        # Обучение v2
│   ├── random_forest.pkl        # Модель v1
│   ├── random_forest_v2.pkl     # Модель v2
│   ├── feature_importance.csv   # Важность признаков v1
│   └── feature_importance_v2.csv# Важность признаков v2
├── app/
│   ├── streamlit_app.py         # Веб-интерфейс (авто-выбор v1/v2)
│   ├── predict_api.py           # CLI API
│   └── visualize_graph.py       # Визуализация графа
├── docs/
│   ├── architecture.md          # Архитектура системы
│   ├── limitations.md           # Ограничения модели
│   └── road_graph.png           # Схема графа
├── compare_models.py            # Сравнение v1 vs v2
├── requirements.txt
└── README.md
```

---

## 🔬 Ограничения

Модель обучена на **симулированных данных**. Для применения в реальном городе необходима калибровка на реальных данных трафика.

Подробнее: [`docs/limitations.md`](docs/limitations.md)

---

## 👥 Команда

Проект создан в рамках хакатона AI inDrive Gov.
