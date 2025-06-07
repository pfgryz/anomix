import numpy as np
from datasets import Dataset, load_from_disk

from anomix.config import PROCESSED_DATA_DIR
from anomix.datasets.definition import DatasetDefinition


def iter_ds(ds: Dataset, label_column: str = "__label__"):
    for x, y in zip(ds.remove_columns(label_column), ds[label_column]):
        x = np.array(list(x.values()), dtype=np.float32)
        yield x, y


def collect_ds(ds: Dataset, label_column: str = "__label__"):
    feature_columns = [col for col in ds.column_names if col != label_column]

    data_dict = ds.to_dict()

    X = np.stack([np.array(data_dict[col], dtype=np.float32) for col in feature_columns], axis=1)
    Y = np.array(data_dict[label_column])

    return X, Y


def load_ds(definition: DatasetDefinition):
    path = PROCESSED_DATA_DIR / f"{definition.name}"
    ds = load_from_disk(str(path), keep_in_memory=True)
    return ds
