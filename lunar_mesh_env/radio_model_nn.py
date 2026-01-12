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

    def _denormalize(self, tensor):
        """Converts a tensor from the [-1, 1] range back to the [0, 1] range."""
        return (tensor.clamp(-1, 1) + 1.0) / 2.0

    def generate_map(self, tx_pos: Tuple[float, float], frequency: str) -> np.ndarray:
        """
        Generates a radiomap for a given transmitter position and frequency.
        """
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
            # print(f"Generated radiomap at tx_pos={tx_pos}, frequency={frequency}")
            return output_dbm
        
        
    def generate_map_batch(self, tx_positions: List[Tuple[float, float]], frequency: str) -> np.ndarray:
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