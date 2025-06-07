import numpy as np
import pytest

from anomix.detection.distribution_based import DistributionBasedDetector


@pytest.fixture
def sample_data():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [2.0, 3.0],
            [5.0, 7.0],
            [12.0, 12.0],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    return X, labels


def test_distribution_based_distances(sample_data):
    X, labels = sample_data

    scorer = DistributionBasedDetector(threshold=1.0)
    scores = scorer.score_samples(X, labels)

    assert scores[2] < 0
    assert scores[2] < scores[0]
    assert scores[2] < scores[1]
    assert scores[5] < 0
    assert scores[5] < scores[3]
    assert scores[5] < scores[4]


def test_distribution_based_detection(sample_data):
    X, labels = sample_data

    scorer = DistributionBasedDetector(threshold=1.0)
    predictions = scorer.fit_predict(X, labels)

    expected = np.array([1, 1, -1, 1, 1, -1])
    assert (predictions == expected).all()


def test_distribution_based_negative_labels(sample_data):
    X, _ = sample_data
    labels = np.array([0, 0, 0, 1, -1, 1])

    scorer = DistributionBasedDetector(threshold=1.0)
    _ = scorer.fit_predict(X, labels)
