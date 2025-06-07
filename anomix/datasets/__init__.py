from datasets import Dataset, load_from_disk

from anomix.config import PROCESSED_DATA_DIR
from anomix.datasets.definition import DatasetDefinition


def iter_ds(ds: Dataset, label_column: str = "__label__"):
    for item in ds:
        y = int(item[label_column])
        x = [v for k, v in item.items() if k != label_column]
        yield x, y


def collect_ds(ds: Dataset, label_column: str = "__label__"):
    return zip(*iter_ds(ds, label_column))


def load_ds(definition: DatasetDefinition):
    path = PROCESSED_DATA_DIR / f"{definition.name}"
    ds = load_from_disk(str(path))
    return ds
