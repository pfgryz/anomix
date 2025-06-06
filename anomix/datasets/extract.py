import shutil
import tarfile
import tempfile
from pathlib import Path

from tqdm import tqdm

from anomix.config import EXTRACTED_DATA_DIR, RAW_DATA_DIR
from anomix.datasets.definition import DatasetDefinition


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    with tarfile.open(archive_path, mode="r:xz") as tar:
        tar.extractall(path=extract_to)


def extract_datasets(definitions: list[DatasetDefinition]) -> None:
    for definitions in tqdm(definitions, desc="Extracting Datasets"):
        destination_path = EXTRACTED_DATA_DIR / f"{definitions.name}.{definitions.format}"
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if destination_path.exists():
            continue

        if definitions.archive is None:
            dataset_path = RAW_DATA_DIR / f"{definitions.name}.{definitions.format}"
            shutil.copy(dataset_path, destination_path)
        else:
            archive_path = RAW_DATA_DIR / f"{definitions.name}.{definitions.archive}"

            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_extracted_path = Path(tmp_dir) / "extract"
                archive_dataset_path = temp_extracted_path / definitions.inner_path

                extract_archive(archive_path, temp_extracted_path)
                shutil.copy(archive_dataset_path, destination_path)
