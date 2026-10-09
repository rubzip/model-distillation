from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplitPaths:
    """Generates all necessary file paths dynamically based on the split name."""
    split_name: str
    base_dir: Path

    @property
    def raw_metadata(self) -> Path:
        return self.base_dir / f"raw/{self.split_name}_metadata.json"

    @property
    def segments_metadata(self) -> Path:
        return self.base_dir / f"processed/{self.split_name}_segments_metadata.json"

    @property
    def segments(self) -> Path:
        return self.base_dir / f"processed/{self.split_name}_segments.h5"

    @property
    def features(self) -> Path:
        return self.base_dir / f"processed/{self.split_name}_features.h5"

    @property
    def raw_embeddings(self) -> Path:
        return self.base_dir / f"processed/{self.split_name}_raw_embeddings.h5"

    @property
    def embeddings(self) -> Path:
        return self.base_dir / f"processed/{self.split_name}_embeddings.h5"
