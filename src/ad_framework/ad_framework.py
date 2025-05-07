import numpy as np


class ADFramework:
    _params: dict

    def __init__(self, **kwargs):
        self._params = kwargs

    def fit(self, X: np.ndarray, **kwargs):
        """Fit estimator to X
        X shape: (n_samples, n_features)
        Returns fitted estimator."""
        return self

    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Perform fit on X and returns labels for X.
        X shape: (n_samples, n_features)
        Returns -1 for outliers and 1 for inliers. Shape: (n_samples,)"""
        pass

    def predict(X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict if a particular sample is an outlier or not.
        X shape: (n_samples, n_features)
        Return shape: (n_samples,) 1 is inliers, -1 outliers"""
        pass

    def score_samples(X: np.ndarray, **kwargs) -> np.ndarray:
        """Anomaly score
        X shape: (n_samples, n_features)
        Return shape: (n_samples,) with score values"""

    def set_params(self, **kwargs):
        self._params = kwargs

    def update_params(self, **kwargs):
        self._params |= kwargs

    def get_params(self, **kwargs) -> dict:
        return self._params
