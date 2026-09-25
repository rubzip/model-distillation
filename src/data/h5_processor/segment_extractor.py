import json
from dataclasses import asdict
from pathlib import Path
from typing import Generator, List, Tuple

import h5py
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from ..metadata import AudioMetadata, AudioSegmentMetadata


class H5SegmentExtractor:
    def __init__(
        self,
        output_h5_path: Path | str,
        output_meta_path: Path | str,
        target_sr: int = 16000,
        segment_steps: int = 16000,
        write_batch_size: int = 512,  # Cuántos segmentos acumular en RAM antes de volcarlos al disco
    ):
        self.output_h5_path = Path(output_h5_path)
        self.output_meta_path = Path(output_meta_path)
        self.target_sr = target_sr
        self.segment_steps = segment_steps
        self.write_batch_size = write_batch_size

        # Crear directorios si no existen
        self.output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_meta_path.parent.mkdir(parents=True, exist_ok=True)

        self.segment_metadata_list: List[AudioSegmentMetadata] = []
        self._global_counter = 0
        self._audio_buffer: List[torch.Tensor] = []
        self._resamplers: dict[int, T.Resample] = {}

    def _load_and_resample(self, filepath: Path | str) -> torch.Tensor:
        """Carga el audio y lo remuestrea si es necesario usando torchaudio o soundfile."""
        try:
            waveform, sr = torchaudio.load(filepath)
        except Exception:
            # Fallback a soundfile para máxima compatibilidad multiplataforma
            data, sr = sf.read(filepath, dtype="float32")
            waveform = torch.from_numpy(data)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

        # Convertir a mono si tiene múltiples canales (promediando)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Remuestrear con resampler en caché
        if sr != self.target_sr:
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = self._resamplers[sr](waveform)

        return waveform

    def _extract_segments(
        self, waveform: torch.Tensor
    ) -> Generator[Tuple[int, int, torch.Tensor], None, None]:
        """Generador que recorta el audio. Si el último segmento es corto, le hace padding con ceros."""
        total_steps = waveform.shape[1]

        for start in range(0, total_steps, self.segment_steps):
            end = start + self.segment_steps
            chunk = waveform[:, start:end]

            # Garantizar que el tamaño sea EXACTAMENTE segment_steps (Padding)
            if chunk.shape[1] < self.segment_steps:
                pad_amount = self.segment_steps - chunk.shape[1]
                chunk = F.pad(chunk, (0, pad_amount))

            yield start, min(end, total_steps), chunk

    def _flush_buffer(self, dataset: h5py.Dataset) -> None:
        """Vuelca el buffer de RAM al disco HDF5 en un solo bloque y lo vacía."""
        if not self._audio_buffer:
            return

        # Apilar todos los tensores del buffer [Batch, 1, Steps] y quitar la dimensión de canal -> [Batch, Steps]
        batch_tensor = torch.cat(self._audio_buffer, dim=0)
        batch_numpy = batch_tensor.cpu().numpy()

        # Expandir el HDF5 y escribir
        current_size = dataset.shape[0]
        dataset.resize((current_size + batch_numpy.shape[0], self.segment_steps))
        dataset[current_size:] = batch_numpy

        # Vaciar buffer
        self._audio_buffer.clear()

    def _save_master_metadata(self) -> None:
        """Serializa la metadata ligera a un único JSON masivo."""
        serialized_list = []
        for meta in self.segment_metadata_list:
            meta_dict = asdict(meta)

            # Serialización segura para JSON
            meta_dict["original_fname"] = str(meta_dict["original_fname"])
            meta_dict["split"] = str(meta_dict["split"])  # StrEnum a string

            serialized_list.append(meta_dict)

        with open(self.output_meta_path, "w", encoding="utf-8") as f:
            json.dump(serialized_list, f, indent=2, ensure_ascii=False)

    def process_file(self, base_meta: AudioMetadata, dataset: h5py.Dataset) -> None:
        """Procesa un solo archivo original y alimenta el buffer/metadata."""
        try:
            waveform = self._load_and_resample(base_meta.fname)

            for start_step, end_step, chunk in self._extract_segments(waveform):
                # El ID es simplemente el índice de fila como string (ej. "0000005")
                segment_id = f"{self._global_counter:07d}"

                # Crear metadato ligero con timestamps calculados y provenance completa
                segment_meta = AudioSegmentMetadata(
                    id=segment_id,
                    original_fname=base_meta.fname,
                    start_step=start_step,
                    end_step=end_step,
                    sr=self.target_sr,
                    split=base_meta.split,
                    category=base_meta.category,
                    subcategory=base_meta.subcategory,
                    location_id=base_meta.location_id,
                    device_id=base_meta.device_id,
                )

                self.segment_metadata_list.append(segment_meta)
                self._audio_buffer.append(chunk)
                self._global_counter += 1

                # Volcar a disco si el buffer está lleno
                if len(self._audio_buffer) >= self.write_batch_size:
                    self._flush_buffer(dataset)

        except Exception as e:
            print(f"Error crítico procesando {base_meta.fname}: {e}")

    def run(self, original_metadata_list: List[AudioMetadata]) -> List[AudioSegmentMetadata]:
        """Orquestador principal."""
        print(f"Iniciando extracción. HDF5: {self.output_h5_path}")

        with h5py.File(self.output_h5_path, "w") as h5f:
            # Crear dataset expansible para el audio con chunks alineados al buffer
            dataset = h5f.create_dataset(
                "audio_segments",
                shape=(0, self.segment_steps),  # Dimensión inicial 0
                maxshape=(
                    None,
                    self.segment_steps,
                ),  # La primera dimensión puede crecer al infinito
                dtype="float32",
                chunks=(self.write_batch_size, self.segment_steps),
                compression="lzf",  # Compresión muy rápida
            )

            for idx, base_meta in enumerate(original_metadata_list):
                self.process_file(base_meta, dataset)

                if (idx + 1) % 50 == 0:
                    print(
                        f"Archivos originales procesados: {idx + 1}/{len(original_metadata_list)}"
                    )

            # Escribir los segmentos restantes que no llenaron el último buffer
            self._flush_buffer(dataset)

            # Guardar atributos de configuración en el HDF5
            h5f.attrs["target_sr"] = self.target_sr
            h5f.attrs["segment_steps"] = self.segment_steps
            h5f.attrs["total_segments"] = self._global_counter

        print(f"Guardando metadata maestra en {self.output_meta_path}...")
        self._save_master_metadata()

        print(f"¡Éxito! Se generaron {self._global_counter} segmentos perfectos.")
        return self.segment_metadata_list
