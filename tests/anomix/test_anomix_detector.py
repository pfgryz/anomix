import numpy as np
import pytest
from sklearn.cluster import KMeans

from anomix.anomix_detector import AnomixDetector
from anomix.detection.cblof import CBLOFDetector


@pytest.fixture
def sample_data():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [10.0, 10.0],
        ]
    )

    return X


def test_anomix_detector_scores(sample_data):
    X = sample_data

    clustering = KMeans(n_clusters=2, random_state=0, n_init="auto")
    detection = CBLOFDetector(threshold=10.0)

    anomix_detector = AnomixDetector(clustering, detection)
    scores = anomix_detector.score_samples(X)

    assert (scores[0:5] > 0).all()
    assert scores[5] < 0


def test_anomix_detector_anomalies(sample_data):
    X = sample_data

    clustering = KMeans(n_clusters=2, random_state=0, n_init="auto")
    detection = CBLOFDetector(threshold=10.0)

    anomix_detector = AnomixDetector(clustering, detection)
    predictions = anomix_detector.fit_predict(X)

    expected = np.array([0, 0, 0, 0, 0, 1])
    assert (predictions == expected).all()
