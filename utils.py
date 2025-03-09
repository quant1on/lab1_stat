import numpy as np
import random as rnd
import math

def get_random_radius(a: float, n: int = 5) -> np.array:
    """Взять рандомные радиусы

    Args:
        a (float): половина стороны квадрата
        n (int, optional): кол-во радиусов (defaults to 5)

    Returns:
        np.array: массив радиусов
    """

    return np.array([rnd.uniform(0.00001, a) for _ in range(n)])

def get_uniform_radius(a: float, n: int = 5) -> np.array:
    """Взять радиусы с одинаковым шагом

    Args:
        a (float): половина стороны квадрата
        n (int, optional): кол-во радиусов (defaults to 5)

    Returns:
        np.array: массив радиусов
    """

    return np.array([(i * a)/n for i in (np.arange(n) + 1)])

def get_true_probability(a: float, r: float) -> float:
    """Посчитать истинную вероятность попадания точки в круг

    Args:
        a (float): половина стороны квадрата
        r (float): радиус круга

    Returns:
        float: вероятность
    """

    r = min(r, a)
    return (r * r * math.pi)/(4 * a * a)

def count_generated_points(a: float, r: float, n: int = 25) -> int:
    """Посчитать, сколько из рандомно сгенерированных точек оказались в круге

    Args:
        a (float): половина стороны квадрата
        r (float): радиус круга
        n (int, optional): кол-во точек (defaults to 25)

    Returns:
        int: кол-во точек, попавших в круг
    """

    r = min(a, r)
    cnt = 0
    for _ in range(n):
        cnt += 1 if np.linalg.norm(np.array([rnd.uniform(0, a) for __ in range(2)])) <= r else 0
    
    return cnt

def get_precision_count(a: float, r: float, err: float = 1e-4, max_iter: int = 1e4, gap: int = 1) -> int:
    """Рассчитать кол-во точек, нужных для выведения истинной вероятности

    Args:
        a (float): половина стороны квадрата
        r (float): радиус круга
        err (float, optional): необходимая точность (defaults to 1e-4)
        max_iter (int, optional): максимальное кол-во итераций вычислений (defaults to 1e4)

    Returns:
        int: Кол-во необходимых точек
    """

    r = min(a, r)
    gap = max(abs(gap), 1)
    true_prob = get_true_probability(a, r)
    N = 1
    while (max_iter > 0):
        prob = count_generated_points(a, r, N) / N

        if abs(true_prob - prob) < err:
            break
        
        N += gap
        max_iter -= 1
        
    return N