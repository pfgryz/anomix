import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from anomix.config import DATASETS_FILE


@dataclass
class DatasetDefinition:
    name: str
    url: str
    format: str
    label_column: str
    anomalous_value: Union[int, str]
    archive: Optional[str] = None
    inner_path: Optional[str] = None


def load_dataset_definitions(path: Path = DATASETS_FILE) -> list[DatasetDefinition]:
    with open(path) as f:
        raw = json.load(f)

    return [DatasetDefinition(**item) for item in raw]
