import json
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import asdict
from typing import List, Generator, Tuple
import numpy as np

from .metadata import AudioMetadata, AudioSegmentMetadata

class SegmentExtractor:
    def __init__(self, output_base_dir: Path, target_sr: int = 16000, segment_steps: int = 16000):
        self.output_base_dir = Path(output_base_dir)
        self.target_sr = target_sr
        self.segment_steps = segment_steps
        
        self.audio_dir = self.output_base_dir / "audio"
        self.meta_dir = self.output_base_dir / "metadata"
        
        self._global_counter = 1
        
        self._setup_directories()

    def _setup_directories(self) -> None:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def _load_and_resample(self, filepath: Path) -> np.ndarray:
        y, _ = librosa.load(filepath, sr=self.target_sr, mono=True)
        return y

    def _extract_segments(self, y: np.ndarray) -> Generator[Tuple[int, int, np.ndarray], None, None]:
        total_steps = len(y)
        
        for start in range(0, total_steps, self.segment_steps):
            end = start + self.segment_steps
            
            if end > total_steps:
                break
                
            yield start, end, y[start:end]

    def _save_audio(self, audio_data: np.ndarray, file_id: str) -> Path:
        wav_filename = f"{file_id}.wav"
        wav_path = self.audio_dir / wav_filename
        sf.write(wav_path, audio_data, self.target_sr)
        return wav_path

    def _save_metadata(self, segment_meta: AudioSegmentMetadata, file_id: str) -> None:
        json_path = self.meta_dir / f"{file_id}.json"
        
        meta_dict = asdict(segment_meta)
        
        meta_dict['fname'] = str(meta_dict['fname'].name)
        if meta_dict['original_fname']:
            meta_dict['original_fname'] = str(meta_dict['original_fname'])
            
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=4, ensure_ascii=False)

    def _create_segment_metadata(
        self, base_meta: AudioMetadata, wav_path: Path, start: int, end: int, file_id: str
    ) -> AudioSegmentMetadata:
        return AudioSegmentMetadata(
            id=file_id,
            fname=wav_path,
            category=base_meta.category,
            subcategory=base_meta.subcategory,
            steps=self.segment_steps,
            sr=self.target_sr,
            channels=1,
            location_id=base_meta.location_id,
            device_id=base_meta.device_id,
            split=base_meta.split,
            original_fname=base_meta.fname,
            start_step=start,
            end_step=end
        )

    def process_file(self, base_meta: AudioMetadata) -> None:
        try:
            y = self._load_and_resample(base_meta.fname)
            
            for start, end, segment_audio in self._extract_segments(y):
                file_id = str(self._global_counter)
                
                # Guardar el audio
                wav_path = self._save_audio(segment_audio, file_id)
                
                # Crear y guardar metadatos
                segment_meta = self._create_segment_metadata(
                    base_meta, wav_path, start, end, file_id
                )
                self._save_metadata(segment_meta, file_id)
                
                self._global_counter += 1
                
        except Exception as e:
            print(f"Error procesando {base_meta.fname}: {e}")

    def run(self, original_metadata_list: List[AudioMetadata]) -> None:
        print(f"Iniciando procesamiento. Guardando en: {self.output_base_dir}")
        
        for base_meta in original_metadata_list:
            self.process_file(base_meta)
            
        print(f"Proceso finalizado. Se han generado {self._global_counter - 1} segmentos.")
