import sys
from pathlib import Path

# Fix python path for script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from config import LABELS, LABELS_DICT, MOCK_PATHS, SIZE_REDUCTION_DIM
from src.data.load_dataset import load_dataset
from src.models import DistillationModel, MLPFactory, VGGishFactory, save_distillation_model
from src.training.criterion import DistillationCriterion
from src.training.early_stopping import EarlyStopping
from src.training.train import train_distillation_model

BATCH_SIZE = 32
PATIENCE = 5
LAMBDA_E = 1.5
LAMBDA_Y = 1.0
DEFAULT_DEVICE = "cuda"
RAISE_IF_NO_DEVICE = False

EMBEDDING_INPUT_SHAPE = (128, 101)
CLASSIFIER_HIDEN_DIMS = [32, 32]
CLASSIFIER_DROPOUT = 0.1

NUM_CLASSES = len(LABELS)

MODEL_NAME = "production_distillation_model.pt"


def main():
    print("Setting up training...")

    # 1. Load Data
    train_paths = MOCK_PATHS["train"]
    val_paths = MOCK_PATHS["val"]

    train_dataset = load_dataset(paths=train_paths, label_map=LABELS_DICT)
    val_dataset = load_dataset(paths=val_paths, label_map=LABELS_DICT)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Setup Model
    print("Instantiating Production Model...")

    # The mel spectrogram shape is (128, 101)
    # Embedding model outputs 32 dimensions (Table 24: FC1 256 ReLU, FC2 32 Linear)
    embedding_model = VGGishFactory.create_vggish200k_mobile(
        input_shape=(128, 101), embed_dim=SIZE_REDUCTION_DIM
    )
    # Classifier model takes 32 inputs and has FC3 (32 ReLU), FC4 (32 ReLU), FC5 (num_classes logits)
    classifier_model = MLPFactory.create_mlp(
        input_size=SIZE_REDUCTION_DIM,
        hidden_sizes=CLASSIFIER_HIDEN_DIMS,
        num_classes=NUM_CLASSES,
        dropout_rate=CLASSIFIER_DROPOUT,
    )
    model = DistillationModel(embedding_model=embedding_model, classifier_model=classifier_model)

    # 3. Setup Training Components
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ea = EarlyStopping(patience=PATIENCE, restore_best_weights=True)

    # From paper: alpha_e = 1.5, alpha_c = 1.0
    criterion = DistillationCriterion(lambda_e=LAMBDA_E, lambda_y=LAMBDA_Y)

    device = torch.device(DEFAULT_DEVICE if torch.cuda.is_available() else "cpu")
    if RAISE_IF_NO_DEVICE and device.type == "cpu":
        raise RuntimeError("CUDA is not available, but RAISE_IF_NO_DEVICE is set to True.")

    print(f"Training on device: {device}")

    # 4. Train
    print("Starting training...")
    history, best_epoch = train_distillation_model(
        model=model,
        ea=ea,
        max_epochs=10,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=NUM_CLASSES,
        device=device,
    )

    print(f"Training finished. Best epoch: {best_epoch}")
    print(f"Final Validation Accuracy: {history.results[-1].accuracy:.4f}")

    print("Saving model...")
    model_save_path = train_paths.models_dir / MODEL_NAME
    save_distillation_model(model, model_save_path)

if __name__ == "__main__":
    main()
