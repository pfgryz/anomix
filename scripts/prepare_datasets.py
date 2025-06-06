import asyncio

from datasets import load_from_disk

from anomix.config import PROCESSED_DATA_DIR
from anomix.datasets.definition import load_dataset_definitions
from anomix.datasets.download import download_datasets
from anomix.datasets.extract import extract_datasets
from anomix.datasets.process import process_datasets


async def main():
    definitions = load_dataset_definitions()
    await download_datasets(definitions)
    extract_datasets(definitions)
    process_datasets(definitions)

    # @TODO: Remove this, placeholder
    # Example on how to load datasets from disk
    for df in definitions:
        pth = PROCESSED_DATA_DIR / f"{df.name}"
        ds = load_from_disk(pth)
        print("Loaded", df.name)
        print(ds)


if __name__ == "__main__":
    asyncio.run(main())
