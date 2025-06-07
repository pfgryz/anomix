import numpy as np
import pytest
from pytest import approx

from anomix.detection.distance_threshold import DistanceThresholdDetector


@pytest.fixture
def sample_data():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [2.0, 1.0],
            [4.0, 5.0],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1])
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
    return X, labels, centroids


def test_distance_treshold_scores(sample_data):
    X, labels, centroids = sample_data

    detector = DistanceThresholdDetector(threshold=0.8)
    scores = detector.score_samples(X, labels, centroids)

    assert scores[0] == approx(0.8)
    assert scores[1] == approx(0.6)
    assert scores[2] == approx(-0.2)
    assert scores[3] == approx(0.8)
    assert scores[4] == approx(-0.2)


@pytest.mark.parametrize(
    "threshold, expected",
    [
        (0.8, (0, 0, 1, 0, 1)),
        (0.1, (0, 1, 1, 0, 1)),
    ],
)
def test_distance_threshold_detection(sample_data, threshold: float, expected: tuple):
    X, labels, centroids = sample_data

    detector = DistanceThresholdDetector(threshold=threshold)
    predictions = detector.fit_predict(X, labels, centroids)

    expected = np.array(expected)
    assert (predictions == expected).all()
