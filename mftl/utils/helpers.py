



from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ModelWeights:

    coefficients: np.ndarray
    intercept: np.ndarray


@dataclass
class TwoTaskWeights:





    coeff_a: np.ndarray
    inter_a: np.ndarray
    coeff_b: np.ndarray
    inter_b: np.ndarray


def _sanitize_vector(vec: np.ndarray) -> np.ndarray:








    v = np.asarray(vec, dtype=float).ravel()
    if not np.all(np.isfinite(v)):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v


def split_tasks_2way(X: np.ndarray, y: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:












    y = np.asarray(y)

    mask_a = (y >= 0) & (y <= 9)
    X_a, y_a = X[mask_a], y[mask_a]

    X_b, y_b = np.array([]).reshape(0, X.shape[1]), np.array([])
    return (X_a, y_a), (X_b, y_b)


def get_classes_for_task(y: np.ndarray) -> np.ndarray:








    classes = np.unique(y)
    return classes if classes.size > 0 else np.arange(10)
