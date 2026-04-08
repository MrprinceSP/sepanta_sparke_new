import sys
sys.path.insert(0, "src")

import os
import torch
import scipy.io.wavfile
from diffusers import AudioLDMPipeline

# Use GPU when available
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device}, torch_dtype: {torch_dtype}")

# Load the AudioLDM pipeline
# Replace this model path if you want a different AudioLDM variant
repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch_dtype)
pipe = pipe.to(device)

prompts = [
    "1920s swing jazz, scratchy vinyl record, upbeat brass section, fast tempo",
    "1950s bebop, high-speed acoustic bass and frantic saxophone improvisation",
    "Modern jazz fusion, electric guitar with chorus effect, tight funk drumming",
    "Smooth lounge jazz, very slow tempo, smoky atmosphere, 1960s recording",
    "Contemporary cinematic jazz, deep reverb, melancholic piano melody"
]

output_dir = "sparke_audio_outputs"
os.makedirs(output_dir, exist_ok=True)

for i, prompt in enumerate(prompts, start=1):
    print(f"Generating audio {i}: {prompt}")
    generator = torch.Generator(device=device).manual_seed(i)
    output = pipe(
        prompt,
        num_inference_steps=30,
        audio_length_in_s=20.0,
        num_waveforms_per_prompt=1,
        generator=generator,
    )

    audio = output.audios[0]
    output_path = os.path.join(output_dir, f"sparke_audio_{i}.wav")
    scipy.io.wavfile.write(output_path, rate=16000, data=audio)
    print(f"Saved {output_path}")

print("Audio generation complete.")
