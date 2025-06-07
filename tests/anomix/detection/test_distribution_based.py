from math import sqrt

import numpy as np
import pytest
from pytest import approx

from anomix.detection.distribution_based import DistributionBasedScorer

@pytest.fixture
def sample_data():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [5.0, 5.0],
        [2.0, 3.0],
        [5.0, 7.0],
        [12.0, 12.0],
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])
    return X, labels


def test_distribution_based_distances(sample_data):
    X, labels = sample_data

    scorer = DistributionBasedScorer(threshold=2.0)
    mean_distances, mu, sigma = scorer.score_samples(X, labels)

    assert mean_distances[0] == approx(2 * sqrt(2))
    assert mean_distances[1] == approx(5/3 * sqrt(2))
    assert mean_distances[2] == approx(3 * sqrt(2))
    assert mean_distances[3] == approx((5 + sqrt(181)) / 3)
    assert mean_distances[4] == approx((5 + sqrt(74)) / 3)
    assert mean_distances[5] == approx((sqrt(181) + sqrt(74)) / 3)


def test_distribution_based_detection(sample_data):
    X, labels = sample_data

    scorer = DistributionBasedScorer(threshold=1.0)
    predictions = scorer.fit_predict(X, labels)

    expected = np.array([0, 0, 1, 0, 0, 1])
    assert (predictions == expected).all()