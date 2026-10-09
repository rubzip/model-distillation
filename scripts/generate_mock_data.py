import json
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from split_paths import PATHS
from dataclasses import dataclass
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

CLASSES = ("class1", "class2", "class3")
SAMPLING_RATES = [16000, 22050, 44100]
LENGTHS_IN_SECONDS = [1.5, 3.0, 5.0]
NUM_CHANNELS = [1, 2]

def generate_mock_audio(meta: AudioMetadata):
    """Generates a dummy sine wave audio file for the metadata."""
    t = np.linspace(0, meta.steps / meta.sr, meta.steps, endpoint=False)
    # Generate a simple 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    if meta.channels == 2:
        audio = np.stack([audio, audio], axis=-1)
        
    meta.fname.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(meta.fname), audio.astype(np.float32), meta.sr)


def get_mock_metadata(i: int, base_path: Path, split: str) -> AudioMetadata:
    fname = base_path / split.lower() / f"audio_{i:04d}.wav"
    category = np.random.choice(CLASSES)
    subcategory = f"{category}_sub"
    length = np.random.choice(LENGTHS_IN_SECONDS)
    sr = int(np.random.choice(SAMPLING_RATES))
    steps = int(sr * length)
    channels = int(np.random.choice(NUM_CHANNELS))
    location_id = f"location_{np.random.randint(1, 10)}"
    device_id = f"device_{np.random.randint(1, 5)}"

    return AudioMetadata(
        fname=fname,
        category=category,
        subcategory=subcategory,
        steps=steps,
        sr=sr,
        channels=channels,
        location_id=location_id,
        device_id=device_id,
        split=split.upper(),
        id=str(uuid.uuid4())[:8],
    )


def generate_dataset(num_train=50, num_val=10, num_test=10):
    splits_config = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }
    
    for split_name, count in splits_config.items():
        print(f"Generating {count} samples for {split_name} split...")
        paths = PATHS[split_name]
        
        metadata_list = []
        root_path = Path(__file__).parent.parent
        for i in range(count):
            meta = get_mock_metadata(i, root_path / "data" / "raw", split_name)
            generate_mock_audio(meta)
            
            # Serialize for JSON
            meta_dict = meta.__dict__.copy()
            meta_dict["fname"] = str(meta.fname)
            metadata_list.append(meta_dict)
            
        paths.raw_metadata.parent.mkdir(parents=True, exist_ok=True)
        with open(paths.raw_metadata, "w") as f:
            json.dump(metadata_list, f, indent=4)
        
        print(f" -> Saved {paths.raw_metadata}")

if __name__ == "__main__":
    np.random.seed(42)
    generate_dataset()
