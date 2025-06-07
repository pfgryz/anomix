import numpy as np
from scipy.spatial.distance import cdist

from anomix.detection import AnomalyDetector


class DistributionBasedDetector(AnomalyDetector):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def score_samples(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray = None) -> np.ndarray:
        mean_distances = np.zeros_like(labels, dtype=float)
        mu = np.zeros_like(labels, dtype=float)
        sigma = np.zeros_like(labels, dtype=float)

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id

            if cluster_id == -1:
                mean_distances[mask] = 10
                mu[mask] = 0
                sigma[mask] = 0
                continue

            x = X[mask]
            D = cdist(x, x)
            mean_distances[mask] = D.mean(axis=1)

            mu[mask] = np.mean(mean_distances)
            sigma[mask] = np.std(mean_distances)

        return mu + self.threshold * sigma - mean_distances
