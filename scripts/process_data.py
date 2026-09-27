from src.data.h5_processor import H5SegmentExtractor
from src.data.metadata import AudioMetadata, AudioSegmentMetadata, load_audio_metadata


def segment_audio(
    raw_audio: list[AudioMetadata],
    output_path: str,
    output_meta_path: str,
    target_sr: int = 16000,
    segment_steps: int = 16000,
) -> list[AudioSegmentMetadata]:
    extractor = H5SegmentExtractor(
        output_h5_path=output_path,
        output_meta_path=output_meta_path,
        target_sr=target_sr,
        segment_steps=segment_steps,
    )
    segments = extractor.run(raw_audio)
    return segments


def run_pipeline():
    # Example placeholder workflow
    train_metadata = load_audio_metadata("data/raw/train_metadata.json")
    val_metadata = load_audio_metadata("data/raw/val_metadata.json")
    test_metadata = load_audio_metadata("data/raw/test_metadata.json")

    extractor = H5SegmentExtractor()
    extractor.run(train_metadata)

    extractor.set_output(
        "data/processed/test_segments.h5", "data/processed/test_segments_metadata.json"
    )
    extractor.run(test_metadata)

    extractor.set_output(
        "data/processed/val_segments.h5", "data/processed/val_segments_metadata.json"
    )
    extractor.run(val_metadata)
