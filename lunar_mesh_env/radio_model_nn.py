# radio_model_nn.py


import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import Normalize
from diffusers import UNet2DModel, DDPMScheduler
from typing import Tuple, Dict, List

from RadioLunaDiff.k2net.diff_modules import K2_UNet
from RadioLunaDiff.pmnet.models.pmnet_v3 import PMNet

from .inference_dataloader import InferenceInputProcessor



class RadioMapModelNN:
    """
    A generative model that runs a 3-stage NN pipeline to predict
    a radiomap based on a transmitter's location and frequency.
    """
    def __init__(self,
                 model_paths: Dict[str, str],
                 heightmap: np.ndarray,
                 env_width: float,
                 env_height: float,
                 num_inference_steps: int = 4,
                 image_size: int = 256,
                 dummy_mode: bool = False,
                 device: str = None):
        
        # print("Initializing Generative RadioMapModel...")
    
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")


        self.cache = {}
        self.image_size = image_size
        self.width = env_width
        self.height = env_height
        
        self.dummy_mode = dummy_mode
        if dummy_mode:
            print("WARNING: RadioMapModelNN running in DUMMY MODE. No real inference will be performed.")
            return

        try:
            # K2 UNet
            self.k2_model = K2_UNet(inputs=4, k2_bin=True).to(self.device)
            self.k2_model.load_state_dict(
                torch.load(model_paths['k2_model'], map_location=self.device)
            )
            
            # PMNet
            self.pmnet = PMNet(
                n_blocks=[3, 3, 27, 3], atrous_rates=[6, 12, 18],
                multi_grids=[1, 2, 4], output_stride=8, in_ch=5
            ).to(self.device)
            self.pmnet.load_state_dict(
                torch.load(model_paths['pmnet_model'], map_location=self.device)
            )
            
            # Diffusion residual
            self.model = UNet2DModel.from_pretrained(
                model_paths['diffusion_model']
            ).to(self.device)

            self.k2_model.eval()
            self.pmnet.eval()
            self.model.eval()
            print("All models loaded successfully.")

        except Exception as e:
            print(f"Error loading models: {e}")
            print("Please check your model_paths dictionary.")
            raise

        
        # noise scheduler for diffusion
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type='v_prediction'
        )
        self.noise_scheduler.set_timesteps(num_inference_steps)
       
        self.normalize_transform = Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5])
        
        self.normalize_transform_single = Normalize(mean=[0.5], std=[0.5])
        
        # print("Preprocessing static heightmaps...")
        # dataloader for deployment
        self.processor = InferenceInputProcessor(image_size=self.image_size)
        
        self.static_hm_tensor = self.processor.preprocess_heightmap(
            heightmap
        ).to(self.device)
        
        self.static_fhm_tensor = self.processor.preprocess_filt_heightmap(
            heightmap
        ).to(self.device)
        
        # print("RadioMapModel initialized and ready.")

    def _get_cache_key(self, x, y, frequency):
        """Standardizes the key for the internal dictionary."""
        return (int(x), int(y), str(frequency))

    def get_signal_strength(self, origin_x:float, origin_y: float, target_x: float, target_y: float, freq: str = '5.8') -> float:
        """
        Calculates signal strength at (tx, ty) from a provided radio_map at pos
        (ox, oy).
        """
        radio_map = self.generate_map((origin_x, origin_y), freq)
        
        if radio_map is None: 
            return -150.0
            
        # map world coords to pixel indices
        r_idx = int((target_y / self.height) * (radio_map.shape[0] - 1))
        c_idx = int((target_x / self.width) * (radio_map.shape[1] - 1))
        
        r_idx = np.clip(r_idx, 0, radio_map.shape[0] - 1)
        c_idx = np.clip(c_idx, 0, radio_map.shape[1] - 1)
        
        return float(radio_map[r_idx, c_idx])

    def _get_cached_map(self, x: float, y: float, frequency: str):
        """Retrieves a map if the quantized position exists in cache."""
        key = (int(x), int(y), frequency)
        return self.cache.get(key, None)

    def _add_to_cache(self, x: float, y: float, frequency: str, radio_map: np.ndarray):
        """Stores a generated map into the internal cache."""
        key = (int(x), int(y), frequency)
        self.cache[key] = radio_map

    def clear_cache(self):
        self.cache = {}
        
    def get_throughput_pos(self, origin_x:float, origin_y:float, target_x:float, target_y:float, freq: str = '5.8') -> float:
        """combines get_signal_strength and get_throughput to determine a throughput based on tx/rx position

        Args:
            origin_x (float): _description_
            origin_y (float): _description_
            target_x (float): _description_
            target_y (float): _description_
            freq (str, optional): _description_. Defaults to '5.8'.

        Returns:
            float: _description_
        """
        rss = self.get_signal_strength(origin_x, origin_y, target_x, target_y, freq)
        return self.get_throughput_rss(rss, freq)
    
    def get_throughput_rss(self, rss:float, frequency_band='5.8') -> float:
        """
        Returns throughput in Mbps based on RSS (dBm).
        Data rate R_data = N_data_bits_per_symbol / T_symbol
        
        N_data_bits_per_symbol = N_data_subcarriers * N_bits_per_sub_carrier * R_coding_rate
        
        Sources:
        - 5.8 GHz: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9363693 PAGE ~2829
        - 415 MHz: placeholder implementation
        """

        if frequency_band == '5.8':
            # From table 17 5, N_data_subcarriers is 48
            N_dsc = 48.0
            
            # T symbol for 20MHz spacing defined in table 17 16
            T_symbol = 4e-6
            
            # N_bit_per_subcarrier is determined by modulation
            # BPSK=1, QPSK=2, 16-QAM=4, 64-QAM=6 used
            N_bpsc = {"BPSK":1.0, "QPSK":2.0, "16-QAM":4.0, "64-QAM":6.0}
            
            # coding rate, modulation, and minimum sensitivity for 20MHz channel spacing
            # defined in table 17-18
            R_cr = [1.0/2.0, 3.0/4.0, 2.0/3.0]
            
            # can be found in  table 17-18
            if rss >= -65: return N_dsc*N_bpsc['64-QAM']*R_cr[1]/T_symbol  # 64-QAM 3/4, 54Mbps
            if rss >= -66: return N_dsc*N_bpsc['64-QAM']*R_cr[2]/T_symbol  # 64-QAM 2/3, 48Mbps
            if rss >= -70: return N_dsc*N_bpsc['16-QAM']*R_cr[1]/T_symbol  # 16-QAM 3/4, 36Mbps
            if rss >= -74: return N_dsc*N_bpsc['16-QAM']*R_cr[0]/T_symbol  # 16-QAM 1/2, 24Mbps
            if rss >= -77: return N_dsc*N_bpsc['QPSK']*R_cr[1]/T_symbol   # QPSK 3/4, 18Mbps
            if rss >= -79: return N_dsc*N_bpsc['QPSK']*R_cr[0]/T_symbol   # QPSK 1/2, 12Mbps
            if rss >= -81: return N_dsc*N_bpsc['BPSK']*R_cr[1]/T_symbol   # BPSK 3/4, 9Mbps
            if rss >= -82: return N_dsc*N_bpsc['BPSK']*R_cr[0]/T_symbol   # BPSK 1/2, 6Mbps
            return 0.0                  # Disconnected

        elif frequency_band == '415':
            if rss >= -85: return 2.0   
            if rss >= -90: return 1.0   
            if rss >= -95: return 0.5   
            if rss >= -105: return 0.1  
            return 0.0

        return 0.0
    
    def _denormalize(self, tensor):
        """Converts a tensor from the [-1, 1] range back to the [0, 1] range."""
        return (tensor.clamp(-1, 1) + 1.0) / 2.0

    def generate_map(self, tx_pos: Tuple[float, float], frequency: str) -> np.ndarray:
        """Wrapper for generate_map_batch to handle single requests via cache."""
        return self.generate_map_batch([tx_pos], frequency)[0]

         
    def generate_map_batch(self, tx_positions: List[Tuple[float, float]], frequency: str) -> np.ndarray:
        """Processes a batch while skipping positions already in the cache."""
        results = [None] * len(tx_positions)
        needed_indices = []
        needed_pos = []

        for i, pos in enumerate(tx_positions):
            key = self._get_cache_key(*pos, frequency)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                needed_indices.append(i)
                needed_pos.append(pos)

        if needed_pos:
            new_maps = self._run_batch_inference(needed_pos, frequency)
            # map generated maps back to original results indices and update cache
            for idx, pos, m in zip(needed_indices, needed_pos, new_maps):
                key = self._get_cache_key(*pos, frequency)
                self.cache[key] = m
                results[idx] = m

        return np.array(results)
    
        
    def _run_batch_inference(self, tx_positions: List[Tuple[float, float]], frequency: str) -> np.ndarray:
        """
        Generates a batch of radiomaps for a list of transmitter positions.
        
        Args:
            tx_positions: List of (x, y) tuples.
            frequency: '5.8' or '415'.
            
        Returns:
            np.ndarray of shape (Batch_Size, Height, Width) containing dBm values.
        """
        if self.dummy_mode:
            batch_maps = []
            for pos in tx_positions:
                Y, X = np.ogrid[:256, :256]
                dist_sq = (X - pos[0])**2 + (Y - pos[1])**2
                dist_sq[dist_sq == 0] = 1.0 
                dummy_dbm = -10 - 20 * np.log10(dist_sq)
                batch_maps.append(np.clip(dummy_dbm, -150, 0))
            return np.array(batch_maps)
        
        batch_size = len(tx_positions)
        if batch_size == 0:
            return np.array([])

        with torch.no_grad():
            # batch tx maps
            tx_maps_list = [
                self.processor.create_tx_map(pos, self.width, self.height, self.device)
                for pos in tx_positions
            ]
            tx_maps = torch.stack(tx_maps_list) 

            # batch freq maps
            freq_map_single = self.processor.create_freq_map(frequency, self.device)
            freq_maps = freq_map_single.unsqueeze(0).expand(batch_size, -1, -1) 

            # expand hms
            static_hm_batch = self.static_hm_tensor.unsqueeze(0).expand(batch_size, -1, -1)
            static_fhm_batch = self.static_fhm_tensor.unsqueeze(0).expand(batch_size, -1, -1)

            # stack inputs: shape (B, 4, 256, 256)
            unnorm_inputs = torch.stack([
                static_hm_batch, 
                tx_maps, 
                freq_maps, 
                static_fhm_batch
            ], dim=1)

            # K2 and PMNet expect inputs in [0, 1]
            k2_inputs = unnorm_inputs.clone()
            pm_inputs = unnorm_inputs.clone()
            
            # only diffusion conditioning maps expect [-1, 1]
            conditioning_maps = self.normalize_transform(unnorm_inputs.clone())
            
            # K2Net 
            k2_map_pred = torch.sigmoid(self.k2_model(k2_inputs))
            
            # PMNet 
            pm_inputs_cat = torch.cat([pm_inputs, k2_map_pred.clone()], dim=1)
            pm_pred = self.pmnet(pm_inputs_cat)

            # normalize PM and K2 maps for diffusion
            pm_pred_norm = self.normalize_transform_single(pm_pred)
            k2_map_norm = self.normalize_transform_single(k2_map_pred)
            
            # Diffusion loop
            generated_images = torch.randn(
                (batch_size, 1, self.image_size, self.image_size),
                device=self.device
            )
            
            for t in self.noise_scheduler.timesteps:
                model_input = torch.cat([
                    generated_images,        
                    conditioning_maps,       
                    pm_pred_norm,            
                    k2_map_norm              
                ], dim=1)

                noise_pred = self.model(model_input, t, return_dict=False)[0]
                
                generated_images = self.noise_scheduler.step(
                    noise_pred, t, generated_images, return_dict=False
                )[0]

            final_gen_norm = pm_pred_norm + generated_images
            final_gen_0_1 = self._denormalize(final_gen_norm)
            
            output_map_0_1 = final_gen_0_1.squeeze(1).cpu().numpy()
            
            # convert back to correct range based on oriinal training
            # dataloaders
            output_dbm = (output_map_0_1 * 210.0) - 200.0 
            # print(f"Generated batch of {batch_size} radiomaps at frequency={frequency}")
            return output_dbm
       
       
       
       
       
       
        
    # DEPRECATED single-map generation method
    def _generate_map(self, tx_pos: Tuple[float, float], frequency: str) -> np.ndarray:
        """
        Generates a radiomap for a given transmitter position and frequency.
        """
        
        key= self._get_cache_key(tx_pos[0], tx_pos[1], frequency)
        if key in self.cache:
            return self._get_cached_map(tx_pos[0], tx_pos[1], frequency)
        
        
        if self.dummy_mode:
            # 1/r^2 for testing purposes
            Y, X = np.ogrid[:256, :256]
            dist_sq = (X - tx_pos[0])**2 + (Y - tx_pos[1])**2
            dist_sq[dist_sq == 0] = 1.0 
            dummy_dbm = -10 - 20 * np.log10(dist_sq)
            return np.clip(dummy_dbm, -150, 0)
        
        
        with torch.no_grad():
        
            tx_map = self.processor.create_tx_map(
                tx_pos, self.width, self.height, self.device
            ) 
            
            freq_map = self.processor.create_freq_map(
                frequency, self.device
            ) 
            
            unnorm_inputs = torch.stack([
                self.static_hm_tensor,     
                tx_map,                    
                freq_map,                  
                self.static_fhm_tensor     
            ]).unsqueeze(0)
            
            k2_inputs = unnorm_inputs.clone()
            pm_inputs = unnorm_inputs.clone()

            conditioning_maps = self.normalize_transform(
                unnorm_inputs.squeeze(0) 
            ).unsqueeze(0)

            # run inference pipeline
  
            k2_map_pred = torch.sigmoid(self.k2_model(k2_inputs))
            
            pm_inputs_cat = torch.cat([pm_inputs, k2_map_pred.clone()], dim=1) 
            pm_pred = self.pmnet(pm_inputs_cat)

            pm_pred_norm = self.normalize_transform_single(pm_pred)
            k2_map_norm = self.normalize_transform_single(k2_map_pred)
            
            generated_images = torch.randn(
                (1, 1, self.image_size, self.image_size),
                device=self.device
            )
            
            for t in self.noise_scheduler.timesteps:
                model_input = torch.cat([
                    generated_images,       
                    conditioning_maps,      
                    pm_pred_norm,           
                    k2_map_norm             
                ], dim=1) 
                
                noise_pred = self.model(model_input, t, return_dict=False)[0]
                generated_images = self.noise_scheduler.step(
                    noise_pred, t, generated_images, return_dict=False
                )[0]

            
            final_gen_norm = pm_pred_norm + generated_images 
            
            final_gen_0_1 = self._denormalize(final_gen_norm)
            
            output_map_0_1 = final_gen_0_1.squeeze().cpu().numpy()
            
            # convert back to correct range based on oriinal training
            # dataloaders
            output_dbm = (output_map_0_1 * 210.0) - 200.0
            self._add_to_cache(tx_pos[0], tx_pos[1], frequency, output_dbm)
            # print(f"Generated radiomap at tx_pos={tx_pos}, frequency={frequency}")
            return output_dbm
        
        