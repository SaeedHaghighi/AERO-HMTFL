



from typing import List, Tuple
import numpy as np
from sklearn.model_selection import train_test_split

from mftl.utils.helpers import split_tasks_2way


def load_minist_data(count_vehicles: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:












    from sklearn.datasets import load_digits


    mnist_datasets = load_digits()
    X_data, y_data = mnist_datasets['data'], mnist_datasets['target']


    (X_task_a, y_task_a), (X_task_b, y_task_b) = split_tasks_2way(X_data, y_data)

    list_data_vehicles_X_data = []
    list_data_vehicles_y_data = []


    samples_per_vehicle_a = len(X_task_a) // count_vehicles if count_vehicles > 0 else 0
    samples_per_vehicle_b = len(X_task_b) // count_vehicles if count_vehicles > 0 else 0

    for i in range(count_vehicles):

        start_a = i * samples_per_vehicle_a
        end_a = (i + 1) * samples_per_vehicle_a if i < count_vehicles - 1 else len(X_task_a)
        X_a_portion = X_task_a[start_a:end_a]
        y_a_portion = y_task_a[start_a:end_a]


        start_b = i * samples_per_vehicle_b
        end_b = (i + 1) * samples_per_vehicle_b if i < count_vehicles - 1 else len(X_task_b)
        X_b_portion = X_task_b[start_b:end_b]
        y_b_portion = y_task_b[start_b:end_b]


        if len(X_a_portion) > 0 and len(X_b_portion) > 0:
            X_combined = np.vstack([X_a_portion, X_b_portion])
            y_combined = np.hstack([y_a_portion, y_b_portion])
        elif len(X_a_portion) > 0:
            X_combined = X_a_portion
            y_combined = y_a_portion
        elif len(X_b_portion) > 0:
            X_combined = X_b_portion
            y_combined = y_b_portion
        else:
            X_combined = np.array([]).reshape(0, X_data.shape[1])
            y_combined = np.array([])

        list_data_vehicles_X_data.append(X_combined)
        list_data_vehicles_y_data.append(y_combined)

    return list_data_vehicles_X_data, list_data_vehicles_y_data, X_data, y_data


def load_data_for_vehicles(num_vehicles: int) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:








    try:

        print("[data] Using load_minist_data()")
        list_X, list_y, all_X, all_y = load_minist_data(count_vehicles=num_vehicles + 1)
        print(f"[data] Loaded project data for {len(list_X)} splits (vehicles+EPC)")
        return list_X, list_y, all_X, all_y

    except Exception as e:
        print(f"[data] Error loading data ({e}); falling back to synthetic data.")


        rng = np.random.RandomState(42)
        all_X = rng.randn(5000, 20)
        all_y = (rng.rand(5000) > 0.5).astype(int)
        list_X = []
        list_y = []
        splits = np.array_split(np.arange(len(all_X)), num_vehicles + 1)

        for idxs in splits:
            list_X.append(all_X[idxs])
            list_y.append(all_y[idxs])

        return list_X, list_y, all_X, all_y
