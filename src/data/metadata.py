from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from enum import StrEnum

class SplitType(StrEnum):
    TRAIN = "TRAIN"
    VAL = "VAL"
    TEST = "TEST"
    NONE = None

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

    # Aplicas el tipo directamente
    split: SplitType

@dataclass
class AudioSegmentMetadata(AudioMetadata):
    original_fname: Optional[Path] = None  
    start_step: Optional[int] = None  
    end_step: Optional[int] = None
