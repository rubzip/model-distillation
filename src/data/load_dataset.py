from config import LABELS_DICT
from src.data.dataset import KnowledgeDistillationDataset
from src.data.metadata import load_segment_metadata
from src.data.split_paths import SplitPaths


def load_dataset(paths: SplitPaths, label_map: dict[str, int]) -> KnowledgeDistillationDataset:
    metadata_list = load_segment_metadata(paths.segments_metadata)

    return KnowledgeDistillationDataset(
        metadata_list=metadata_list,
        features_h5_path=paths.features,
        embeddings_h5_path=paths.embeddings,
        label_map=label_map,
    )
