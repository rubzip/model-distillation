# Acoustic Scene Classification (ASC) - Model Distillation

Framework for data processing and knowledge distillation in Acoustic Scene Classification for hearing aids, based on my Master's Thesis: [Acoustic Scene Classification in Hearing Aids for Noise Reduction Applications](https://rubzip.github.io/publications/noise/) ([Full Thesis PDF](https://drive.google.com/file/d/1PKXW68vWbPL_5RZLggep4g8vjV1vAQkG/view)).

## The Problem & Solution

Hearing aids need to identify acoustic environments in real time to adapt noise suppression algorithms dynamically. However, embedded hearing aid hardware operates under strict compute constraints (requiring < 20 MFLOPs). Pretrained audio embedding models like VGGish are too heavy for this use case (783+ MFLOPs, ~275 MB).

To solve this:
1. **Model Miniaturization**: In my thesis, we compressed the model down to 1.6 MFLOPs and ~0.45 MB using Depthwise Separable Convolutions, dimensionality reduction, and knowledge distillation.
2. **This Repository**: Provides the modular pipeline to take raw audio files, prepare the distilled training dataset, and train the production student model:
   - Slices raw audios into 1-second segments.
   - Extracts log-mel spectrogram features.
   - Precomputes teacher audio embeddings (128-dim).
   - Reduces embedding dimensionality using Incremental PCA (down to 32-dim).
   - Trains the lightweight student model using a simultaneous loss function ($\alpha_e = 1.5$ embedding RMSE + $\alpha_c = 1.0$ classification cross-entropy).

## Why HDF5 (`.h5`) Files?

Processing hundreds of hours of audio generates massive volumes of segments, spectrograms, and embeddings:
- **RAM Limits**: Storing all tensors in memory causes out-of-memory errors on typical machines.
- **I/O Overhead**: Saving millions of tiny files introduces OS open/close bottlenecks.
- **Solution**: Intermediate features and embeddings are stored in structured HDF5 files. The PyTorch `Dataset` loads slices lazily from disk per batch, keeping RAM usage constant regardless of dataset size.

## Project Structure

```
├── config.py                 # Central configuration and paths
├── mock_data/                # Synthetic data for local testing
├── scripts/
│   ├── data/                 # Data creation, processing, and validation
│   └── training/             # Model training scripts
├── src/
│   ├── data/                 # HDF5 processors and PyTorch Dataset
│   ├── models/               # Architectures (VGGish, MLP, Production)
│   ├── processing/           # Mel spectrogram, VGGish, and PCA extractors
│   └── training/             # Loss criteria, metrics, early stopping, and loop
└── tests/                    # Unit and integration tests
```

## Running the Pipeline

All dependencies and commands are managed with `uv`.

### 1. Generate Mock Data
Generates synthetic audio files and raw metadata:
```bash
uv run python scripts/data/generate_mock_data.py
```

### 2. Process Dataset
Extracts segments, computes mel spectrograms, generates embeddings, and fits PCA:
```bash
uv run python scripts/data/process_data.py
```

### 3. Validate Dataset
Checks dataset integrity and verifies shapes across splits:
```bash
uv run python scripts/data/validate_dataset.py --mock
```

### 4. Train the Production Model
Trains the distilled lightweight mobile model:
```bash
uv run python scripts/training/train_production_model.py
```

### 5. Run Tests
Runs the test suite across processors, datasets, models, and training:
```bash
./test.sh
```
