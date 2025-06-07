import numpy as np

from anomix.detection import AnomalyDetector


class CBLOFDetector(AnomalyDetector):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def score_samples(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        cluster_sizes = np.bincount(labels)
        assigned_centroids = centroids[labels]

        distances = np.linalg.norm(X - assigned_centroids, axis=1)
        cluster_weights = cluster_sizes[labels]
        cblof_scores = distances * cluster_weights

        return self.threshold - cblof_scores
