from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import StrEnum

class SplitType(StrEnum):
    TRAIN = "TRAIN"
    VAL = "VAL"
    TEST = "TEST"
    NONE = "NONE"

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

@dataclass
class AudioSegmentMetadata:
    id: str
    original_fname: Path | str
    start_step: int
    end_step: int
    sr: int
    
    split: SplitType
    category: str
