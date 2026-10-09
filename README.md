# Acoustic Scene Clasification Model Distillation
This repo contains a framework for implementing my Masters thesis work (it doesnt contain data nor specific architectures).

## Useful Commands (`uv`)

Below are the main commands for working with this project using `uv`:

### 1. Generate Synthetic Data (Mock Data)
Generates dummy `.wav` files and their respective metadata in `mock_data/raw/`:
```bash
uv run python scripts/generate_mock_data.py
```

### 2. Process the Dataset
Runs the complete processing pipeline (segment extraction, MEL feature generation, and dimensionality reduction with PCA):
```bash
uv run python scripts/process_data.py
```

### 3. Run the Tests
To run the unit and validation test suite:
```bash
uv run pytest tests/
```
