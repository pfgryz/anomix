from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class AnomalyDetector(BaseEstimator):
    @abstractmethod
    def score_samples(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray: ...

    def fit_predict(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray = None) -> np.ndarray:
        scores = self.score_samples(X, labels, centroids)
        return (scores < 0).astype(int)
