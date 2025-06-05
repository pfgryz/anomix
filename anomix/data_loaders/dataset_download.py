from dataclasses import dataclass
from pathlib import Path
from .dataset_wrapper import (
    ARFFCategoricalDatasetWrapper,
    CSVNumericalDatasetWrapper,
    FileDatasetWrapperBase,
    TarXZArchiveNumericalDatasetWrapper,
)
import asyncio
import aiohttp
from ..config import DATASET_RAW
from tqdm import tqdm


@dataclass
class DownloadableDatasetDefinition:
    download_url: str
    destination_path: Path
    handler: FileDatasetWrapperBase


CHUNK_SIZE = 1024


async def download_file(session, dataset_def: DownloadableDatasetDefinition, dataset_progress_bar: tqdm):
    async with session.get(dataset_def.download_url) as response:
        with open(dataset_def.destination_path, "wb") as f:
            async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                if not chunk:
                    break
                f.write(chunk)

        dataset_def.handler.preprocess_file(dataset_def.destination_path)
        dataset_progress_bar.update(1)


DATASETS = [
    DownloadableDatasetDefinition(
        download_url="https://raw.githubusercontent.com/GuansongPang/ADRepository-Anomaly-detection-datasets/refs/heads/main/numerical%20data/DevNet%20datasets/KDD2014_donors_10feat_nomissing_normalised.csv",
        destination_path=DATASET_RAW / "donors.csv",
        handler=CSVNumericalDatasetWrapper("donors", "class", 1),
    ),
    DownloadableDatasetDefinition(
        download_url="https://raw.githubusercontent.com/GuansongPang/ADRepository-Anomaly-detection-datasets/refs/heads/main/numerical%20data/DevNet%20datasets/annthyroid_21feat_normalised.csv",
        destination_path=DATASET_RAW / "thyroid.csv",
        handler=CSVNumericalDatasetWrapper("thyroid", "class", 1),
    ),
    DownloadableDatasetDefinition(
        download_url="https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/refs/heads/main/numerical%20data/DevNet%20datasets/UNSW_NB15_traintest_backdoor.tar.xz",
        destination_path=DATASET_RAW / "UNSW_NB15_traintest_backdoor.tar.xz",
        handler=TarXZArchiveNumericalDatasetWrapper("backdoors", "class", 1),
    ),
    DownloadableDatasetDefinition(
        download_url="https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/refs/heads/main/numerical%20data/DevNet%20datasets/bank-additional-full_normalised.csv",
        destination_path=DATASET_RAW / "bank.csv",
        handler=CSVNumericalDatasetWrapper("bank", "class", 1),
    ),
    DownloadableDatasetDefinition(
        download_url="https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/refs/heads/main/numerical%20data/DevNet%20datasets/celeba_baldvsnonbald_normalised.csv",
        destination_path=DATASET_RAW / "celeba.csv",
        handler=CSVNumericalDatasetWrapper("celeba", "class", 1),
    ),
    DownloadableDatasetDefinition(
        download_url="https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/refs/heads/main/numerical%20data/DevNet%20datasets/census-income-full-mixed-binarized.tar.xz",
        destination_path=DATASET_RAW / "census-income-full-mixed-binarized.tar.xz",
        handler=TarXZArchiveNumericalDatasetWrapper("census", "class", 1),
    ),
    # DownloadableDatasetDefinition(
    #     download_url="https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/refs/heads/main/numerical % 20data/DevNet % 20datasets/creditcardfraud_normalised.tar.xz",
    #     destination_path=DATASET_RAW / "census-income-full-mixed-binarized.tar.xz",
    #     handler=TarXZArchiveNumericalDatasetWrapper("fraud", "class", 1)
    # ), 300MB
    DownloadableDatasetDefinition(
        download_url="https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/refs/heads/main/categorical%20data/bank-additional-ful-nominal.arff",
        destination_path=DATASET_RAW / "bank.arff",
        handler=ARFFCategoricalDatasetWrapper("bank", "y", "yes"),
    ),
    DownloadableDatasetDefinition(
        download_url="https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/refs/heads/main/categorical%20data/solar-flare_FvsAll-cleaned.arff",
        destination_path=DATASET_RAW / "flare.arff",
        handler=ARFFCategoricalDatasetWrapper("flare", "class", 1),
    ),
    DownloadableDatasetDefinition(
        download_url="https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets/raw/refs/heads/main/categorical%20data/chess_krkopt_zerovsall.arff",
        destination_path=DATASET_RAW / "chess.arff",
        handler=ARFFCategoricalDatasetWrapper("chess", "class", 1),
    ),
]


async def cache_datasets(num_workers: int = 3):
    async with aiohttp.ClientSession() as session:
        datasets_to_download = [ds for ds in DATASETS if not ds.destination_path.exists()]

        dataset_progress_bar = tqdm(total=len(datasets_to_download), desc="Datasets")

        semaphore = asyncio.Semaphore(num_workers)

        async def worker(ds):
            async with semaphore:
                await download_file(session, ds, dataset_progress_bar)

        await asyncio.gather(*(worker(ds) for ds in datasets_to_download))
        dataset_progress_bar.close()


def load_datasets():
    for dataset in DATASETS:
        if not dataset.destination_path.exists():
            raise Exception("{dataset.name} is not downloaded, please run the download script first")

    for dataset in DATASETS:
        print(f"Loading {dataset.handler.name}")
        yield dataset.handler.from_file(dataset.destination_path)
