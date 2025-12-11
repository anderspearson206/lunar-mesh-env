# inference_dataloader.py

import numpy as np
import torch
import scipy.ndimage
from torchvision import transforms
from typing import Tuple

# these values taken from the original dataset/dataloaders
HM_GLOBAL_MIN = 0
HM_GLOBAL_MAX = 496
FHM_GLOBAL_MIN = 0
FHM_GLOBAL_MAX = 140

class InferenceInputProcessor:
    """
    Handles the precise data preprocessing (normalization, fhm calculation,
    resizing, etc.) to match the RefractLunarLoader settings used
    during model training.
    """
    def __init__(self, 
                 image_size: int = 256,
                 hm_norm_params: Tuple[float, float] = (HM_GLOBAL_MIN, HM_GLOBAL_MAX),
                 fhm_norm_params: Tuple[float, float] = (FHM_GLOBAL_MIN, FHM_GLOBAL_MAX)
                ):
        
        self.image_size = image_size
        
        # heightmap norm
        self.hm_min, self.hm_max = hm_norm_params
        self.hm_norm_range = self.hm_max - self.hm_min
        if self.hm_norm_range == 0:
            self.hm_norm_range = 1.0

        # fhm norm
        self.fhm_min, self.fhm_max = fhm_norm_params
        self.fhm_norm_range = self.fhm_max - self.fhm_min
        if self.fhm_norm_range == 0:
            self.fhm_norm_range = 1.0

        self.resize = transforms.Resize((image_size, image_size), antialias=True)

    def _normalize(self, data: np.ndarray, min_val: float, norm_range: float) -> np.ndarray:
        """Applies global min-max normalization and clips to [0, 1]."""
        return np.clip((data.astype(np.float32) - min_val) / norm_range, 0, 1)

    def preprocess_heightmap(self, hm: np.ndarray) -> torch.Tensor:
        """
        Applies the two-step normalization from the dataloader:
        1. Local min subtraction
        2. Global min-max scaling
        """
        hm_local_norm = hm - np.min(hm)
        hm_global_norm = self._normalize(hm_local_norm, self.hm_min, self.hm_norm_range)
        
        hm_tensor = torch.from_numpy(hm_global_norm).float().unsqueeze(0) 
        hm_resized = self.resize(hm_tensor).squeeze(0) 
        return hm_resized

    def preprocess_filt_heightmap(self, hm: np.ndarray) -> torch.Tensor:
        """
        Calculates the Filtered Heightmap (FHM) using the provided formula
        and applies the correct two-step normalization.
        """
        # get the fhm
        fhm = hm - scipy.ndimage.gaussian_filter(np.copy(hm), sigma=45)
        
        fhm_local_norm = fhm - np.min(fhm)
        
        fhm_global_norm = self._normalize(fhm_local_norm, self.fhm_min, self.fhm_norm_range)
        
        fhm_tensor = torch.from_numpy(fhm_global_norm).float().unsqueeze(0) 
        fhm_resized = self.resize(fhm_tensor).squeeze(0) 
        return fhm_resized

    def create_tx_map(self, 
                      tx_pos: Tuple[float, float], 
                      env_width: float, 
                      env_height: float, 
                      device: torch.device) -> torch.Tensor:
        """
        Creates a 2D tensor with a single '1' at the transmitter's
        pixel location.
        """
        tx_x_idx = int((tx_pos[0] / env_width) * (self.image_size - 1))
        tx_y_idx = int((tx_pos[1] / env_height) * (self.image_size - 1))
        
        tx_x_idx = np.clip(tx_x_idx, 0, self.image_size - 1)
        tx_y_idx = np.clip(tx_y_idx, 0, self.image_size - 1)
        
        tx_map = torch.zeros((self.image_size, self.image_size), device=device)
        tx_map[tx_y_idx, tx_x_idx] = 1.0 
        return tx_map

    def create_freq_map(self, frequency: str, device: torch.device) -> torch.Tensor:
        """
        Creates a 2D tensor filled with the frequency value.
        '5.8' -> 1.0
        '415' -> 0.0
        """
        freq_val = 1.0 if frequency == '5.8' else 0.0
        freq_map = torch.full(
            (self.image_size, self.image_size), 
            freq_val, 
            device=device
        )
        return freq_map