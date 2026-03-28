import numpy as np
import os

def generate_od_matrix(graph, city_type="mixed"):
    """
    Генерирует матрицу поездок (Origin-Destination)
    
    city_type:
    - "center": центр города (все едут везде)
    - "residential": спальный район (поездки в центр)
    - "mixed": смешанный тип
    """
    n = graph.number_of_nodes()
    matrix = np.zeros((n, n))
    
    if city_type == "center":
        # центр: много поездок между всеми узлами
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = np.random.randint(50, 200)
                    
    elif city_type == "residential":
        # спальный район: поездки в центр (первые 3 узла) и обратно
        center_nodes = [0, 1, 2]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if j in center_nodes and i not in center_nodes:
                    # из спального в центр
                    matrix[i][j] = np.random.randint(50, 150)
                elif i in center_nodes and j not in center_nodes:
                    # из центра в спальный
                    matrix[i][j] = np.random.randint(30, 100)
                elif i not in center_nodes and j not in center_nodes:
                    # между спальными
                    matrix[i][j] = np.random.randint(5, 30)
                else:
                    matrix[i][j] = np.random.randint(10, 60)
                    
    else:  # mixed
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = np.random.randint(10, 100)
    
    return matrix

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.generate_graph import load_graph
    
    # загружаем граф
    G = load_graph()
    
    # создаём OD-матрицу
    od = generate_od_matrix(G, "mixed")
    
    print(f"✅ OD-матрица создана")
    print(f"   Размер: {od.shape}")
    print(f"   Всего поездок: {int(od.sum())}")
    print(f"   Среднее: {od[od > 0].mean():.1f} поездок на маршрут")
    
    # сохраняем
    np.save("od_matrix.npy", od)
    print(f"💾 Сохранена в data/od_matrix.npy")