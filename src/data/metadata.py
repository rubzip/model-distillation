from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from enum import Enum

class SplitType(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

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
