default: run

run:
    uv run python main.py

format:
    ruff check
    ruff format --fix