from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplitPaths:
    """Generates all necessary file paths dynamically based on the split name."""

    split_name: str
    base_dir: Path

    metadata_name: str = "metadata.json"
    metadata_path: str = "raw"
    segments_metadata_name: str = "segments_metadata.json"
    segments_metadata_path: str = "processed"
    segments_name: str = "segments.h5"
    segments_path: str = "processed"
    features_name: str = "features.h5"
    features_path: str = "processed"
    raw_embeddings_name: str = "raw_embeddings.h5"
    raw_embeddings_path: str = "processed"
    embeddings_name: str = "embeddings.h5"
    embeddings_path: str = "processed"
    models_dir_name: str = "models"

    @property
    def raw_metadata(self) -> Path:
        return self.base_dir / self.metadata_path / f"{self.split_name}_{self.metadata_name}"

    @property
    def segments_metadata(self) -> Path:
        return (
            self.base_dir
            / self.segments_metadata_path
            / f"{self.split_name}_{self.segments_metadata_name}"
        )

    @property
    def segments(self) -> Path:
        return self.base_dir / self.segments_path / f"{self.split_name}_{self.segments_name}"

    @property
    def features(self) -> Path:
        return self.base_dir / self.features_path / f"{self.split_name}_{self.features_name}"

    @property
    def raw_embeddings(self) -> Path:
        return (
            self.base_dir
            / self.raw_embeddings_path
            / f"{self.split_name}_{self.raw_embeddings_name}"
        )

    @property
    def embeddings(self) -> Path:
        return self.base_dir / self.embeddings_path / f"{self.split_name}_{self.embeddings_name}"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / self.models_dir_name
