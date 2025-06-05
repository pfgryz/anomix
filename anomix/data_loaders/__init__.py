from .trivial import get_blob_datasets
from .dataset_wrapper import DatasetWrapperBase
from .dataset_download import load_datasets


def get_all_datasets() -> list[DatasetWrapperBase]:
    return [
        *get_blob_datasets(),
        *load_datasets(),
        # get_cardiotocography(),
        # get_forest_cover(),
        # get_kddcup99()
    ]
