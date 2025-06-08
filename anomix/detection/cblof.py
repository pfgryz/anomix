import numpy as np

from anomix.detection import AnomalyDetector


class CBLOFDetector(AnomalyDetector):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def score_samples(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        mask = labels != -1
        cluster_sizes = np.bincount(labels[mask].astype(int))

        assigned_centroids = centroids[labels[mask]]
        distances = np.linalg.norm(X[mask] - assigned_centroids, axis=1)
        cluster_weights = cluster_sizes[labels[mask]]

        cblof_scores = np.ones(X.shape[0]) * 1000
        cblof_scores[mask] = distances * cluster_weights

        return self.threshold - cblof_scores
