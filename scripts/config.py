from src.processing.embedding.vggish import VGGishProcessor
from src.processing.feature_extraction.mel import MelProcessor
from src.processing.size_reduction.pca import PCAEmbeddingReducer

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

CLASSES = ("class1", "class2", "class3")
CLASS_DICT = {cls: idx for idx, cls in enumerate(CLASSES)}
