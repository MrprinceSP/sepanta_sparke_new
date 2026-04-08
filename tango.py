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
        
        print("DEBUG: Starting Tango init")
        
        path = snapshot_download(repo_id=name)
        
        vae_config = json.load(open("{}/vae_config.json".format(path)))
        stft_config = json.load(open("{}/stft_config.json".format(path)))
        main_config = json.load(open("{}/main_config.json".format(path)))
        
        # Load the UNet config directly from local file
        unet_config_path = os.path.join(os.path.dirname(__file__), 'tango-master', 'tango2', 'configs', 'diffusion_model_config.json')
        with open(unet_config_path, 'r') as f:
            unet_config = json.load(f)
        
        # Remove metadata keys that shouldn't be passed to constructor
        unet_config_clean = {k: v for k, v in unet_config.items() if not k.startswith('_')}
        
        print("DEBUG: unet_config loaded, len:", len(unet_config))
        print("DEBUG: unet_config_clean len:", len(unet_config_clean))
        print("DEBUG: sample keys:", list(unet_config.keys())[:5])
        
        main_config["unet_model_config"] = unet_config_clean
        if "unet_model_config_path" in main_config:
            del main_config["unet_model_config_path"]  # Remove the path key entirely
        main_config["unet_model_config_path"] = None
        
        print("DEBUG: unet_config_clean keys:", list(unet_config_clean.keys()))
        print("DEBUG: main_config unet_model_config keys:", list(main_config.get("unet_model_config", {}).keys()))
        
        print("DEBUG: main_config keys:", list(main_config.keys()))
        print("DEBUG: main_config['unet_model_config'] is not None:", main_config.get("unet_model_config") is not None)
        print("DEBUG: type of main_config['unet_model_config']:", type(main_config.get("unet_model_config")))
        
        self.vae = AutoencoderKL(**vae_config).to(device)
        self.stft = TacotronSTFT(**stft_config).to(device)
        print("DEBUG: About to call AudioDiffusion")
        # Pass parameters explicitly
        self.model = AudioDiffusion(
            text_encoder_name=main_config["text_encoder_name"],
            scheduler_name=main_config["scheduler_name"],
            unet_model_config=unet_config_clean,
            snr_gamma=main_config.get("snr_gamma"),
            freeze_text_encoder=main_config.get("freeze_text_encoder", True),
            uncondition=main_config.get("uncondition", False)
        ).to(device)
        
        vae_weights = torch.load("{}/pytorch_model_vae.bin".format(path), map_location=device)
        stft_weights = torch.load("{}/pytorch_model_stft.bin".format(path), map_location=device)
        main_weights = torch.load("{}/pytorch_model_main.bin".format(path), map_location=device)
        
        self.vae.load_state_dict(vae_weights)
        self.stft.load_state_dict(stft_weights)
        self.model.load_state_dict(main_weights)

        print ("Successfully loaded checkpoint from:", name)
        
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