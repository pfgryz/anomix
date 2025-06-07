import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator


class DistributionBasedScorer(BaseEstimator):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def score_samples(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        mean_distances = np.zeros_like(labels, dtype=float)
        mu = np.zeros_like(labels, dtype=float)
        sigma = np.zeros_like(labels, dtype=float)

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            x = X[mask]

            D = cdist(x, x)
            mean_distances[mask] = D.mean(axis=1)

            mu[mask] = np.mean(mean_distances)
            sigma[mask] = np.std(mean_distances)

        return mean_distances, mu, sigma

    def fit_predict(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        mean_distances, mu, sigma = self.score_samples(X, labels)

        return (mean_distances > mu + self.threshold * sigma).astype(int)
