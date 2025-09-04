import numpy as np


def is_dominated(point, population):
    return np.any(np.all(population >= point, axis=1) & np.any(population > point, axis=1))
