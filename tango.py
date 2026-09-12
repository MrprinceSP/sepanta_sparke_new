import json
import sys
import os
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from diffusers import DDPMScheduler

# Add tango-master to path for imports
tango_master_path = os.path.join(os.path.dirname(__file__), 'tango-master')
sys.path.insert(0, os.path.join(tango_master_path, 'tango2'))
sys.path.insert(0, tango_master_path)

from models import AudioDiffusion
from audioldm.audio.stft import TacotronSTFT
from audioldm.variational_autoencoder import AutoencoderKL

class Tango:
    def __init__(self, name="declare-lab/tango", device="cuda:0"):
        import tempfile
        import json
        import os
        
        print("DEBUG: Starting Tango init")
        
        path = snapshot_download(repo_id=name)
        
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        main_config = json.load(open("{}/main_config.json".format(path)))
        
        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        
        # 1. Clean up old/unused keys
        main_config.pop("unet_model_config", None)
        
        # 2. Locate UNet config dict dynamically
        unet_dict = None
        
        # Check inside snapshot first
        snapshot_config_file = os.path.join(path, "diffusion_model_config.json")
        if os.path.exists(snapshot_config_file):
            with open(snapshot_config_file, "r") as f:
                unet_dict = json.load(f)
        else:
            # Check local tango-master repository fallback
            local_unet_path = os.path.join(
                os.path.dirname(__file__), 
                "tango-master", "tango2", "configs", "diffusion_model_config.json"
            )
            if os.path.exists(local_unet_path):
                with open(local_unet_path, "r") as f:
                    unet_dict = json.load(f)

        if unet_dict is None:
            raise FileNotFoundError("Could not locate UNet configuration file.")

        # Filter private metadata keys if any exist
        unet_dict_clean = {k: v for k, v in unet_dict.items() if not k.startswith('_')}

        # 3. Create a clean temporary directory with config.json for diffusers
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_config_path = os.path.join(self.temp_dir.name, "config.json")
        
        with open(temp_config_path, "w") as f:
            json.dump(unet_dict_clean, f)
            
        main_config["unet_model_config_path"] = self.temp_dir.name
        
        print("DEBUG: About to call AudioDiffusion")
        self.model = AudioDiffusion(**main_config).to(device)
        
        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)
        main_weights = torch.load("{}/pytorch_model_main.bin".format(path), map_location=device)
        
        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        print("Successfully loaded checkpoint from:", name)
        
        self.vae.eval()
        self.stft.eval()
        self.model.eval()
        
        self.scheduler = DDPMScheduler.from_pretrained(main_config["scheduler_name"], subfolder="scheduler")
        
    def chunks(self, lst, n):
        """ Yield successive n-sized chunks from a list. """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
        
    def generate(self, prompt, steps=100, guidance=3, samples=1, disable_progress=True):
        """ Genrate audio for a single prompt string. """
        with torch.no_grad():
            latents = self.model.inference([prompt], self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
            mel = self.vae.decode_first_stage(latents)
            wave = self.vae.decode_to_waveform(mel)
        return wave[0]
    
    def generate_for_batch(self, prompts, steps=100, guidance=3, samples=1, batch_size=8, disable_progress=True):
        """ Genrate audio for a list of prompt strings. """
        outputs = []
        for k in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[k: k+batch_size]
            with torch.no_grad():
                latents = self.model.inference(batch, self.scheduler, steps, guidance, samples, disable_progress=disable_progress)
                mel = self.vae.decode_first_stage(latents)
                wave = self.vae.decode_to_waveform(mel)
                outputs += [item for item in wave]
        if samples == 1:
            return outputs
        else:
            return list(self.chunks(outputs, samples))