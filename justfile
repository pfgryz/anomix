default: prepare-env

prepare-env:
    uv sync

run:
    uv run python main.py

format:
    uv run ruff check
    uv run ruff format

clean-datasets:
    rm -rf datasets/*

download-datasets:
    uv run python scripts/download_datasets.py