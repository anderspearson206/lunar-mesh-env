# radio_model.py (old)

import numpy as np
from typing import List, Dict, Tuple

# NOTE: This entire class is supposed to simulate our NN until it is finished training



class RadioMapModel:
    """
    A mock model that simulates the generative radiomap model.
    It loads a fixed dataset of pre-computed radiomaps and returns
    the map corresponding to the closest known transmitter position.
    """
    def __init__(self, data_points: List[Dict]):
        """
        Initializes the model by loading all available map data.
        
        Args:
            data_points: A list of dictionaries, where each dict contains:
                         {'id': str, 'pos': (x, y), '5.8': 'path/to/5.8.npy', '415': 'path/to/415.npy'}
        """
        self.tx_positions = {} 
        self.maps = {} 

        for dp in data_points:
            pos = dp['pos']
            tx_id = dp['id']
            self.tx_positions[tx_id] = pos
            self.maps[tx_id] = {}

            if dp.get('5.8'):
                try:
                    self.maps[tx_id]['5.8'] = np.nan_to_num(np.load(dp['5.8']), False, -110,-110, -110)
                except Exception as e:
                    print(f"Warning: Could not load 5.8GHz map for {tx_id}: {e}")
                    self.maps[tx_id]['5.8'] = None
            
            if dp.get('415'):
                try:
                    self.maps[tx_id]['415'] = np.nan_to_num(np.load(dp['415']), False, -110,-110, -110)
                except Exception as e:
                    print(f"Warning: Could not load 415MHz map for {tx_id}: {e}")
                    self.maps[tx_id]['415'] = None
        
        print(f"RadioMapModel initialized with {len(self.tx_positions)} data points.")

    def generate_map(self, tx_pos: Tuple[float, float], frequency: str) -> np.ndarray:
        """
        Finds the closest known TX position to the requested
        tx_pos and returns its pre-computed map for the given frequency.
        
        Args:
            tx_pos: The (x, y) position of the desired transmitter.
            frequency: The frequency key, e.g., '5.8' or '415'.
            
        Returns:
            The corresponding radiomap as a NumPy array, or None.
        """
        if not self.tx_positions:
            return None

        closest_id = None
        min_dist = np.inf
        
        req_pos_arr = np.array(tx_pos)
        for tx_id, pos in self.tx_positions.items():
            dist = np.linalg.norm(req_pos_arr - np.array(pos))
            if dist < min_dist:
                min_dist = dist
                closest_id = tx_id
        
        if (closest_id 
            and frequency in self.maps[closest_id] 
            and self.maps[closest_id][frequency] is not None):
            
            return self.maps[closest_id][frequency]
        else:
            print(f"Warning: No map found for {closest_id} at freq {frequency}")
            return None