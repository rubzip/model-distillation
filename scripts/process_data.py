import torch
from torch.utils.data import DataLoader
import h5py

from src.data.h5_processor import (
    H5SegmentExtractor,
    H5FeatureProcessor,
    H5EmbeddingProcessor,
    H5SizeReductionProcessor,
)
from src.data.metadata import load_metadata
from scripts.config import MEL_PROCESSOR, EMBEDDING_PROCESSOR, SIZE_REDUCTION_PROCESSOR
from scripts.split_paths import PATHS, SplitPaths

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_pipeline():
    extract_segments()
    extract_features()
    process_embddings()
    train_size_reduction()
    size_reduction()

def extract_segments():
    for split_name, paths in PATHS.items():
        if not paths.raw_metadata.exists():
            raise
        
        print(f"Loading {split_name} metadata")
        metadata = load_metadata(paths.raw_metadata)
        extractor = H5SegmentExtractor(
            output_h5_path=paths.segments, 
            output_meta_path=paths.segments_metadata
        )
        print(f"Extracting {split_name} segments")
        extractor.run(metadata)

def extract_features():
    for split_name, paths in PATHS.items():
        if not paths.segments.exists():
            raise

        print(f"Extracting {split_name.upper()} features")
        feat_processor = H5FeatureProcessor(
            input_h5_path=paths.segments,
            output_h5_path=paths.features,
            processor=MEL_PROCESSOR,
            device=DEVICE,
            batch_size=512
        )
        feat_processor.run()

def process_embddings():
    for split_name, paths in PATHS.items():
        if not paths.segments.exists():
            continue

        print(f"Generating embeddings for: {split_name.upper()}")
        emb_processor = H5EmbeddingProcessor(
            input_h5_path=paths.segments,
            output_h5_path=paths.raw_embeddings,
            processor=EMBEDDING_PROCESSOR,
            device=DEVICE,
            batch_size=256
        )
        emb_processor.run()

def train_size_reduction():
    train_paths = PATHS["train"]
    if not train_paths.raw_embeddings.exists():
        raise

    with h5py.File(train_paths.raw_embeddings, "r") as f:
        train_embeddings_np = f["data"][:]
    
    train_tensor = torch.from_numpy(train_embeddings_np)
    train_loader = DataLoader(train_tensor, batch_size=1024, shuffle=True)
    
    SIZE_REDUCTION_PROCESSOR.fit(train_loader)


def size_reduction():
    for split_name, paths in PATHS.items():
        if not paths.raw_embeddings.exists():
            raise

        print(f"Embedding size reduction {split_name.upper()}")
        red_processor = H5SizeReductionProcessor(
            input_h5_path=paths.raw_embeddings,
            output_h5_path=paths.embeddings,
            reducer=SIZE_REDUCTION_PROCESSOR,
            device=DEVICE,
            batch_size=512
        )
        red_processor.run()    

if __name__ == "__main__":
    run_pipeline()
