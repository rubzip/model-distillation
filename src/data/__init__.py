from .metadata import AudioMetadata, AudioSegmentMetadata, SplitType
from .dataset import KnowledgeDistillationDataset
from .h5_processor import (
    AbstractH5BatchProcessor,
    H5SegmentExtractor,
    H5FeatureProcessor,
    H5EmbeddingProcessor,
    H5SizeReductionProcessor,
    PCAEmbeddingReducer,
)
from .validation import (
    DatasetValidator,
    validate_dataset,
    check_split_leakage,
)

__all__ = [
    "AudioMetadata",
    "AudioSegmentMetadata",
    "SplitType",
    "KnowledgeDistillationDataset",
    "AbstractH5BatchProcessor",
    "H5SegmentExtractor",
    "H5FeatureProcessor",
    "H5EmbeddingProcessor",
    "H5SizeReductionProcessor",
    "PCAEmbeddingReducer",
    "DatasetValidator",
    "validate_dataset",
    "check_split_leakage",
]
