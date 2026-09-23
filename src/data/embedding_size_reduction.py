import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from pathlib import Path

class PCAEmbeddingReducer:
    def __init__(self, target_dim: int = 32):
        self.target_dim = target_dim
        self.pca = IncrementalPCA(n_components=target_dim)
        self.is_fitted = False

    def fit(self, train_loader: DataLoader, embedding_index: int = 0) -> None:        
        for batch in train_loader:
            embeddings = batch[embedding_index]            
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()                
            self.pca.partial_fit(embeddings)
            
        self.is_fitted = True

    def get_retained_variance(self)-> np.ndarray:            
        return self.pca.explained_variance_ratio_

    def transform(self, e: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("El PCA no está entrenado. Llama a fit() primero.")
            
        if isinstance(e, torch.Tensor):
            e = e.cpu().numpy()
            
        return self.pca.transform(e)
