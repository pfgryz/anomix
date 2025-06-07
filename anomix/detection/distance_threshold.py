import numpy as np
from sklearn.base import BaseEstimator


class DistanceThresholdScorer(BaseEstimator):

    def __init__(self, threshold: float):
        self.threshold = threshold

    def score_samples(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        assigned_centroids = centroids[labels]

        distances = np.linalg.norm(X - assigned_centroids, axis=1)
        distances_norm = np.zeros_like(distances)

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id

            d = distances[mask]
            if d.max() == d.min():
                distances_norm[mask] = 0.0
            else:
                distances_norm[mask] = (d - d.min()) / (d.max() - d.min())

        return distances_norm

    def fit_predict(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        scores = self.score_samples(X, labels, centroids)
        return (scores > self.threshold).astype(int)
