import numpy as np
from datasets import Dataset
from sklearn.datasets import make_blobs, make_moons


def create_blobs_dataset(centers: list[tuple[float, float]], cluster_std: float, n_samples: int, label: int) -> Dataset:
    X, _ = make_blobs(centers=centers, cluster_std=cluster_std, n_samples=n_samples, random_state=0)
    y = np.full((n_samples,), label)
    return Dataset.from_dict(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "__label__": y,
        }
    )


def create_moons_dataset(
    scale: float, noise: float, offset: tuple[float, float], n_samples: int, label: int
) -> Dataset:
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=0)
    X = scale * (X - offset)
    y = np.full((n_samples,), label)
    return Dataset.from_dict(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "__label__": y,
        }
    )


def create_uniform_anomalies_dataset(low: float, high: float, n_samples: int, label: int) -> Dataset:
    rng = np.random.RandomState(0)
    X = rng.uniform(low=low, high=high, size=(n_samples, 2))
    y = np.full((n_samples,), label)
    return Dataset.from_dict(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "__label__": y,
        }
    )
