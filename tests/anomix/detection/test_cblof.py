import numpy as np
import pytest

from anomix.detection.cblof import CBLOFScorer


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

    scorer = CBLOFScorer(threshold=0.0)
    scores = scorer.score_samples(X, labels, centroids)

    assert scores[2] > scores[1]  # 1st closer to centroid
    assert scores[5] > scores[4]  # 4th closer to centroid
    assert scores[5] > scores[3]  # 3th closer to centroid
    assert scores[3] > scores[4]  # 4th closer to centroid


def test_cblof_anomaly_detection(sample_data):
    X, labels, centroids = sample_data

    scorer = CBLOFScorer(threshold=10.0)
    predictions = scorer.fit_predict(X, labels, centroids)

    expected = np.array([0, 0, 0, 0, 0, 1])
    assert (predictions == expected).all()
