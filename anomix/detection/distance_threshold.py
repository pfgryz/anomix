import numpy as np

from anomix.detection import AnomalyDetector


class DistanceThresholdDetector(AnomalyDetector):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def score_samples(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        mask = labels != -1

        assigned_centroids = np.zeros_like(X)
        assigned_centroids[mask] = centroids[labels[mask]]

        distances = np.linalg.norm(X - assigned_centroids, axis=1)
        distances_norm = np.zeros_like(distances)

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id

            if cluster_id == -1:
                distances_norm[mask] = 1
                continue

            d = distances[mask]
            if d.max() == d.min():
                distances_norm[mask] = 0.0
            else:
                distances_norm[mask] = (d - d.min()) / (d.max() - d.min())

        return self.threshold - distances_norm
