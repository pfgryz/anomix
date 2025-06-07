import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

from anomix.detection import AnomalyDetector


class AnomixDetector(BaseEstimator):
    def __init__(self, clustering_method: ClusterMixin, detection_method: AnomalyDetector):
        self._clustering_method = clustering_method
        self._detection_method = detection_method

    def fit(self, X: np.ndarray):
        self._labels = self._clustering_method.fit_predict(X)

        if hasattr(self._clustering_method, "cluster_centers_"):
            self._centroids = self._clustering_method.cluster_centers_
        else:
            clusters = np.unique(self._labels)
            self._centroids = np.array([X[self._labels == c].mean(axis=0) for c in clusters])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._labels is None or self._centroids is None:
            raise RuntimeError("You must call fit() before predict().")

        self._predictions = self._detection_method.fit_predict(X, self._labels, self._centroids)
        return self._predictions

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self._detection_method.score_samples(X, self._labels, self._centroids)
