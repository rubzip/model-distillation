uv sync
uv run python scripts/generate_mock_data.py
uv run python scripts/process_data.py
uv run python -m pytest tests/
