uv sync
uv run python scripts/data/generate_mock_data.py
uv run python scripts/data/process_data.py
uv run python -m pytest tests/
