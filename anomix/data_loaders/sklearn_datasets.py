import numpy as np
from .dataset_wrapper import DatasetWrapperBase
from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_covtype
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split


def get_kddcup99():
    X, y = fetch_kddcup99(subset="SA", percent10=True, random_state=42, return_X_y=True, as_frame=True)
    y = (y != b"normal.").astype(np.int32)
    X, _, y, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)
    return DatasetWrapperBase("kddcup99", X, y)


def get_forest_cover():
    X, y = fetch_covtype(return_X_y=True, as_frame=True)
    s = (y == 2) + (y == 4)
    X = X.loc[s]
    y = y.loc[s]
    y = (y != 2).astype(np.int32)

    X, _, y, _ = train_test_split(X, y, train_size=0.05, stratify=y, random_state=42)
    return DatasetWrapperBase("forest_cover", X, y)


def get_cardiotocography():
    X, y = fetch_openml(name="cardiotocography", version=1, return_X_y=True, as_frame=False)
    s = y == "3"
    y = s.astype(np.int32)
    return DatasetWrapperBase("cardiotocography", X, y)
