import numpy as np
from sklearn.datasets import make_blobs, make_moons


# region @TODO: Refactor to common code with anomix.datasets
def combine_datasets(datasets):
    """
    Combine multiple (X, Y) datasets into a single dataset.

    Parameters:
        datasets (list of tuples): Each element is a tuple (X, Y) where:
            - X is an (n_samples, n_features) array
            - Y is an (n_samples,) or (n_samples, 1) array of binary labels (0 or 1)

    Returns:
        (X_combined, Y_combined): Concatenated X and Y arrays
    """
    X_parts = []
    Y_parts = []

    for X, Y in datasets:
        X_parts.append(X)
        Y_parts.append(Y.reshape(-1, 1))  # ensure column vector

    X_combined = np.concatenate(X_parts, axis=0)
    Y_combined = np.concatenate(Y_parts, axis=0)
    return X_combined, Y_combined


def create_blobs_data(centers, cluster_std, n_samples, sample_classes):
    """
    Generate blob-based data.
    """
    X, _ = make_blobs(centers=centers, cluster_std=cluster_std, n_samples=n_samples, random_state=0)
    return X, np.ones((n_samples)) * sample_classes


def create_moons_data(n_samples, scale, noise, offset, sample_classes):
    """
    Generate scaled moon-shaped data.
    """
    X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=0)
    return scale * (X - offset), np.ones((n_samples)) * sample_classes


def create_uniform_anomalies(n_samples, low, high, sample_classes):
    """
    Generate uniform random noise as outliers.
    """
    rng = np.random.RandomState(0)
    return rng.uniform(low=low, high=high, size=(n_samples, 2)), np.ones((n_samples)) * sample_classes


# def get_blob_datasets() -> list[DatasetWrapperBase]:
#     return [
#         DatasetWrapperBase(
#             "one_blob_and_noise_2d",
#             *combine_datasets(
#                 [
#                     create_blobs_data([(0, 0)], 1.0, 100, 0),
#                     create_uniform_anomalies(20, -5, 5, 1),
#                 ]
#             ),
#         ),
#         DatasetWrapperBase(
#             "two_blobs_and_noise",
#             *combine_datasets(
#                 [
#                     create_blobs_data([(-0.5, -0.5), (0.5, 0.5)], 1.0, 100, 0),
#                     create_uniform_anomalies(20, -5, 5, 1),
#                 ]
#             ),
#         ),
#         DatasetWrapperBase(
#             "two_moons",
#             *combine_datasets(
#                 [
#                     create_moons_data(100, 4, 0.2, [0.2, 0.3], 0),
#                     create_blobs_data([(0.5, 0.5)], 1.0, 100, 0),
#                     create_uniform_anomalies(20, -5, 5, 1),
#                 ]
#             ),
#         ),
#     ]
# endregion
