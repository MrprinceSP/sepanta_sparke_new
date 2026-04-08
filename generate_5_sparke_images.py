import sys
sys.path.insert(0, 'src')

import torch
import logging
import clip
from diffusers import SPARKEGuidedStableDiffusionXLPipeline

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch_dtype = torch.float16 if device == 'cuda' else torch.float32

print(f"Using device: {device}, torch_dtype: {torch_dtype}")
from diffusers.pipelines.rke_guidance_utils import RKEGuidedSampling

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load CLIP model for guidance
clip_for_guidance, pre_process_clip = clip.load("ViT-B/32", device=device)

# Load the SPARKE Guided Stable Diffusion XL Pipeline
# Note: Replace the model path with your actual model path
pipe = SPARKEGuidedStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",  # Use a standard SDXL model
    torch_dtype=torch_dtype,
).to(device)

# Set up RKE Guided Sampling
rke_guided_sampler = RKEGuidedSampling(
    algorithm="cond-rke",
    kernel="gaussian",
    sigma_image=0.6,
    sigma_text=0.3,
    max_bank_size=1000,
    use_latents_for_guidance=True,
    model_name="sdxl",
    model=pipe
)

# Guidance parameters
guidance_kwargs = dict(
    guidance_scale=7.5,
    criteria="vscore_clip",
    algorithm="vscore_clip",
    criteria_guidance_scale=0.02,
    guidance_freq=10,
    F_M=None,
    F_T=None,
    F_M_real=None,
    F_T_real=None,
    beta=0,
    regions_list=[],
    return_kernels=False,
    rke_guided_sampler=rke_guided_sampler,
    logger_=logger,
    clip_for_guidance=clip_for_guidance,
    regularize=False,
    regularize_weight=0,
)

# Prompts for generating 5 images
prompts = [
    "A serene mountain landscape at dawn",
    "A futuristic city with flying cars",
    "A portrait of a cat wearing sunglasses",
    "An underwater scene with colorful fish",
    "A vintage car in a desert"
]

# Generate and save 5 images
for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}: {prompt}")
    out = pipe(
        prompt=prompt + ", 4K, realistic",
        negative_prompt="blurry, low-quality, distorted",
        generator=torch.Generator(device=device).manual_seed(i),  # Different seed for each
        num_inference_steps=50,
        **guidance_kwargs
    )
    image = out[0][0]
    image.save(f"sparke_output_{i+1}.png")
    print(f"Saved sparke_output_{i+1}.png")

print("All 5 images generated successfully!")