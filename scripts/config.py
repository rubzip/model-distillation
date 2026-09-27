from src.processing.base import BaseProcessing

SAMPLING_RATE = 16000
SEGMENT_STEPS = 16000

TRAIN_RAW_METADATA_PATH = "data/raw/train_metadata.json"
VAL_RAW_METADATA_PATH = "data/raw/val_metadata.json"
TEST_RAW_METADATA_PATH = "data/raw/test_metadata.json"

TRAIN_SEGMENTED_PATH = "data/processed/train_segments.h5"
VAL_SEGMENTED_PATH = "data/processed/val_segments.h5"
TEST_SEGMENTED_PATH = "data/processed/test_segments.h5"

TRAIN_SEGMENTED_METADATA_PATH = "data/processed/train_segments_metadata.json"
VAL_SEGMENTED_METADATA_PATH = "data/processed/val_segments_metadata.json"
TEST_SEGMENTED_METADATA_PATH = "data/processed/test_segments_metadata.json"

TRAIN_FEATURES_PATH = "data/processed/train_features.h5"
VAL_FEATURES_PATH = "data/processed/val_features.h5"
TEST_FEATURES_PATH = "data/processed/test_features.h5"

TRAIN_RAW_EMBEDDINGS_PATH = "data/processed/train_raw_embeddings.h5"
VAL_RAW_EMBEDDINGS_PATH = "data/processed/val_raw_embeddings.h5"
TEST_RAW_EMBEDDINGS_PATH = "data/processed/test_raw_embeddings.h5"

TRAIN_EMBEDDINGS_PATH = "data/processed/train_embeddings.h5"
VAL_EMBEDDINGS_PATH = "data/processed/val_embeddings.h5"
TEST_EMBEDDINGS_PATH = "data/processed/test_embeddings.h5"

EMBEDDING_CLS: BaseProcessing = ...
SIZE_REDUCTION_CLS: BaseProcessing = ...

CLASSES = ("class1", "class2", "class3")
CLASS_DICT = {cls: idx for idx, cls in enumerate(CLASSES)}
