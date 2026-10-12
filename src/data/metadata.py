import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

SplitType = Literal["TRAIN", "VAL", "TEST", "NONE"]


@dataclass
class AudioMetadata:
    fname: Path
    category: str
    subcategory: str

    steps: int
    sr: int
    channels: int

    location_id: str
    device_id: str
    split: SplitType
    id: Optional[str] = None


def load_metadata(fname: Path) -> list[AudioMetadata]:
    with open(fname, "r") as f:
        metadata_list = json.load(f)
    return [AudioMetadata(**item) for item in metadata_list]


def load_segment_metadata(fname: Path) -> list["AudioSegmentMetadata"]:
    with open(fname, "r") as f:
        metadata_list = json.load(f)
    return [AudioSegmentMetadata(**item) for item in metadata_list]


@dataclass
class AudioSegmentMetadata:
    id: str
    original_fname: Path | str
    start_step: int  # Sample index in resampled domain (at target sr)
    end_step: int  # Sample index in resampled domain (at target sr)
    sr: int  # Target sample rate of the segment

    split: SplitType
    category: str
    subcategory: Optional[str] = None
    location_id: Optional[str] = None
    device_id: Optional[str] = None
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None

    def __post_init__(self):
        if self.start_time_sec is None and self.sr > 0:
            self.start_time_sec = round(self.start_step / self.sr, 6)
        if self.end_time_sec is None and self.sr > 0:
            self.end_time_sec = round(self.end_step / self.sr, 6)
