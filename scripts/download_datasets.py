import asyncio

from anomix.data_loaders.dataset_download import cache_datasets


def main():
    asyncio.run(cache_datasets())


if __name__ == "__main__":
    main()
