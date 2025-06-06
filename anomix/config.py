from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATASETS_FILE = PROJ_ROOT / "datasets.json"

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTRACTED_DATA_DIR = DATA_DIR / "extracted"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INPUT_DATA_DIR = DATA_DIR / "input"
