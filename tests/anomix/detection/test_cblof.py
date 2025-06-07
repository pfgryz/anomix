import numpy as np
import pytest

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
    labels = np.array([0, 0, 0, 1, 1, 1])
    centroids = np.array(
        [
            [0.1, -0.0333],
            [6.7, 6.7],
        ]
    )
    return X, labels, centroids


def test_cblof_scores(sample_data):
    X, labels, centroids = sample_data

    detector = CBLOFDetector(threshold=10.0)
    scores = detector.score_samples(X, labels, centroids)

    assert scores[2] < scores[1]  # 2md closer to centroid
    assert scores[5] < scores[4]  # 5th closer to centroid
    assert scores[5] < scores[3]  # 4th closer to centroid
    assert scores[3] < scores[4]  # 5th closer to centroid
    assert scores[5] < 0  # 6th is anomaly


def test_cblof_anomaly_detection(sample_data):
    X, labels, centroids = sample_data

    detector = CBLOFDetector(threshold=10.0)
    predictions = detector.fit_predict(X, labels, centroids)

    expected = np.array([1, 1, 1, 1, 1, -1])
    assert (predictions == expected).all()


def test_cblof_negative_labels(sample_data):
    X, _, centroids = sample_data
    labels = np.array([0, 0, 0, 1, -1, 1])

    detector = CBLOFDetector(threshold=10.0)
    _ = detector.fit_predict(X, labels, centroids)
