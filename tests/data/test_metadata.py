import unittest
from pathlib import Path

from src.data.metadata import AudioMetadata, AudioSegmentMetadata


class TestMetadata(unittest.TestCase):
    def test_audio_metadata_creation(self):
        meta = AudioMetadata(
            fname=Path("/tmp/audio.wav"),
            category="airport",
            subcategory="terminal",
            steps=160000,
            sr=16000,
            channels=2,
            location_id="loc_1",
            device_id="dev_a",
            split="TRAIN",
            id="audio_001",
        )
        self.assertEqual(meta.category, "airport")
        self.assertEqual(meta.split, "TRAIN")
        self.assertEqual(meta.sr, 16000)

    def test_audio_segment_metadata_timestamp_calculation(self):
        seg = AudioSegmentMetadata(
            id="0000010",
            original_fname=Path("/tmp/audio.wav"),
            start_step=16000,
            end_step=32000,
            sr=16000,
            split="VAL",
            category="park",
        )
        self.assertEqual(seg.id, "0000010")
        self.assertEqual(seg.start_time_sec, 1.0)
        self.assertEqual(seg.end_time_sec, 2.0)
        self.assertEqual(seg.split, "VAL")

    def test_audio_segment_metadata_custom_timestamp(self):
        seg = AudioSegmentMetadata(
            id="0000001",
            original_fname="/tmp/audio.wav",
            start_step=500,
            end_step=1500,
            sr=16000,
            split="TEST",
            category="bus",
            start_time_sec=0.03125,
            end_time_sec=0.09375,
        )
        self.assertEqual(seg.start_time_sec, 0.03125)
        self.assertEqual(seg.end_time_sec, 0.09375)


if __name__ == "__main__":
    unittest.main()
