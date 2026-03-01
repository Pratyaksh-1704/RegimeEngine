import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class WindowedDataset(Dataset):
    """
    Creates dataset of consecutive feature windows for self-supervised learning.
    Returns pairs of windows (w1, w2) which are positive pairs (adjacent in time).
    """
    def __init__(self, features_df: pd.DataFrame, window_size: int = 21):
        """
        Args:
            features_df: DataFrame containing features, sorted by time.
            window_size: Lookback window size.
        """
        self.features = features_df.values
        self.window_size = window_size
        
        # Valid indices where we can extract two windows: x_t and x_{t+1}
        # x_t needs indices [i : i + window_size]
        # x_{t+1} needs indices [i + 1 : i + 1 + window_size]
        # So maximum i is len(features) - window_size - 1
        self.valid_indices = np.arange(len(self.features) - self.window_size - 1)
        
    def __len__(self):
        return len(self.valid_indices)
        
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        
        # Shape: (window_size, num_features)
        w1 = self.features[start_idx : start_idx + self.window_size]
        w2 = self.features[start_idx + 1 : start_idx + 1 + self.window_size]
        
        # Return as (num_features, window_size) for Conv1d
        w1_tensor = torch.tensor(w1, dtype=torch.float32).transpose(0, 1)
        w2_tensor = torch.tensor(w2, dtype=torch.float32).transpose(0, 1)
        
        return w1_tensor, w2_tensor

class InferenceDataset(Dataset):
    """
    Creates dataset of windows for inference (extracting the latent space z_t).
    """
    def __init__(self, features_df: pd.DataFrame, window_size: int = 21):
        self.features = features_df.values
        self.window_size = window_size
        self.valid_indices = np.arange(len(self.features) - self.window_size + 1)
        # Store dates for matching predictions to timeline
        self.dates = features_df.index[self.window_size - 1:]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        w = self.features[start_idx : start_idx + self.window_size]
        w_tensor = torch.tensor(w, dtype=torch.float32).transpose(0, 1)
        return w_tensor
