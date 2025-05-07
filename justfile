default: run

run:
    uv run python main.py

format:
    ruff check
    ruff format --fix

clean_datasets:
    rm -rf datasets/*

download_datasets:
    uv run -m src.data_loaders.dataset_download