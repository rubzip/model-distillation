import sys
from pathlib import Path

# Fix python path for script execution
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import LABELS_DICT, MOCK_PATHS, PATHS, SPLITS
from src.data import load_dataset, validate_dataset


def load_and_validate_dataset(paths):
    dataset = load_dataset(paths, label_map=LABELS_DICT)
    validate_dataset(dataset)


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Load and validate dataset")
    parser.add_argument("--mock", action="store_true", help="Use mock dataset paths")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    paths = MOCK_PATHS if args.mock else PATHS
    for split in SPLITS:
        print(f"Loading and validating {split} dataset...")
        load_and_validate_dataset(paths[split])
    print("All datasets loaded and validated successfully.")
