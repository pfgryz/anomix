import asyncio

from anomix.datasets.definition import load_dataset_definitions
from anomix.datasets.download import download_datasets
from anomix.datasets.extract import extract_datasets
from anomix.datasets.process import process_datasets


async def main():
    definitions = load_dataset_definitions()
    await download_datasets(definitions)
    extract_datasets(definitions)
    process_datasets(definitions)


if __name__ == "__main__":
    asyncio.run(main())
