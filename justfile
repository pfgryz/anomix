default: prepare-env

# region Environment

prepare-env:
    uv sync

clean-datasets:
    rm -rf data/*

download-datasets:
    uv run python scripts/prepare_datasets.py

# endregion

# region Developer Tools

format:
    uv run ruff check --fix
    uv run ruff format

test:
    uv run -m pytest

# endregion

# region Experiments

run:
    uv run python main.py
    
# endregion