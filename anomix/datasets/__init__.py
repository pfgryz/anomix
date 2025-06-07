from datasets import Dataset


def iter_ds(ds: Dataset, label_column: str = "__label__"):
    for item in ds:
        y = int(item[label_column])
        x = [v for k, v in item.items() if k != label_column]
        yield x, y


def collect_ds(ds: Dataset, label_column: str = "__label__"):
    return zip(*iter_ds(ds, label_column))
