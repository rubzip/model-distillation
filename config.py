from pathlib import Path

from src.processing.embedding.vggish import VGGishProcessor
from src.processing.feature_extraction.mel import MelProcessor
from src.processing.size_reduction.pca import PCAEmbeddingReducer

BASE_DIR = Path(__file__).parent / "data"
BASE_DIR_MOCK_DATA = Path(__file__).parent / "mock_data"

SPLITS = ["train", "val", "test"]


# ==========================================
# 2. AUDIO & PROCESSING HYPERPARAMETERS
# ==========================================

SAMPLING_RATE: int = 16000
SEGMENT_STEPS: int = 16000

N_MELS: int = 128
TARGET_SR: int = SAMPLING_RATE
N_FFT: int = 400
HOP_LENGTH: int = 160

EMBEDING_DIM = 128
SIZE_REDUCTION_DIM = 32

MEL_PROCESSOR = MelProcessor(N_MELS, TARGET_SR, N_FFT, HOP_LENGTH)
EMBEDDING_PROCESSOR = VGGishProcessor(target_dim=EMBEDING_DIM)
SIZE_REDUCTION_PROCESSOR = PCAEmbeddingReducer(target_dim=SIZE_REDUCTION_DIM)

# ==========================================
# 3. DATASET CLASSES
# ==========================================

LABELS = ("class1", "class2", "class3")
LABELS_DICT = {cls: idx for idx, cls in enumerate(LABELS)}


from src.data.split_paths import SplitPaths

PATHS: dict[str, SplitPaths] = {
    split_name: SplitPaths(split_name=split_name, base_dir=BASE_DIR)
    for split_name in SPLITS
}

MOCK_PATHS: dict[str, SplitPaths] = {
    split_name: SplitPaths(split_name=split_name, base_dir=BASE_DIR_MOCK_DATA)
    for split_name in SPLITS
}
