import numpy as np
from sklearn.base import BaseEstimator


class CBLOFScorer(BaseEstimator):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def score_samples(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        cluster_sizes = np.bincount(labels)
        assigned_centroids = centroids[labels]

        distances = np.linalg.norm(X - assigned_centroids, axis=1)
        cluster_weights = cluster_sizes[labels]
        cblof_scores = distances * cluster_weights

        return cblof_scores

    def fit_predict(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X, labels, centroids)
        return (scores > self.threshold).astype(int)
