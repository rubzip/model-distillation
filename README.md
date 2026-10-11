# Acoustic Scene Clasification Model Distillation
This repository contains a framework for implementing my Masters thesis work (it doesnt contain data nor specific architectures).
Link to paper.

## Dataset Pipeline
This code goes from taking a group of audio files to building a fully functional dataset for training similar pipeline as masters work.
Basically, it is based on distillation of a pretrained audio embedding model meanwhile training a classifier architecture.
The focus of the work was building a low resource model so the pipeline:
 1. Segment all audio files into fixed-size audio segments.
 2. Precompute audio embeddings
 3. Reduce dimensionality of audio embeddings
 4. Precompute feature extraction algorithm (as Mel Spectrogram)

## Why h5 files
 1. Keeping different files adds a lot of opening closing delay
 2. No ram over

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
