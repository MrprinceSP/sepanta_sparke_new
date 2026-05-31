import sys
sys.path.insert(0, "src")

import os
import torch
import numpy as np # <--- Added numpy for audio saving
import logging
import scipy.io.wavfile

# 1. IMPORT YOUR MODIFIED PIPELINE AND RKE UTILS
from diffusers.pipelines.audioldm2.pipeline_audioldm2 import AudioLDM2Pipeline # <--- Correct Import


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# CRITICAL FIX: Force float32 to prevent the VAE from overflowing into pure static
torch_dtype = torch.float32 

print(f"Using device: {device}, torch_dtype: {torch_dtype}")

# 2. LOAD THE AUDIOLDM2 PIPELINE
repo_id = "cvssp/audioldm2-music" # <--- Updated to AudioLDM2 Music Model
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device) # <--- Updated class name





# 5. PROMPTS TO GENERATE
prompts = [
    "1920s swing jazz, scratchy vinyl record, upbeat brass section, fast tempo",
    "1950s bebop, high-speed acoustic bass and frantic saxophone improvisation",
    "Modern jazz fusion, electric guitar with chorus effect, tight funk drumming",
    "Smooth lounge jazz, very slow tempo, smoky atmosphere, 1960s recording",
    "Contemporary cinematic jazz, deep reverb, melancholic piano melody",
    "Solo jazz piano, complex chords, stride style, no other instruments",
    "Jazz drum solo, focus on cymbals and snares, rhythmic complexity",
    "Acoustic jazz trio, double bass, piano, and brushed drums",
    "Big band jazz, massive horn section, loud and energetic",
    "Solo gypsy jazz guitar, fast picking, acoustic wood body sound",
    "Dark, noir jazz, solo trumpet with heavy reverb, rainy street atmosphere",
    "Upbeat Bossa Nova, nylon string guitar, light percussion, sunny feel",
    "Avant-garde free jazz, dissonant piano, chaotic rhythms, experimental",
    "Soul-jazz, groovy Hammond organ, bluesy electric guitar",
    "Ethereal vocal jazz, scat singing, soft background accompaniment",
    "Jazz band playing in a massive stone cathedral, long echo tails",
    "Intimate jazz club recording, sounds of clinking glasses and distant chatter",
    "Street performer playing a saxophone in a subway tunnel, metallic echoes",
    "Jazz quintet, very wide stereo separation, professional studio mastering",
    "Low-fidelity bedroom jazz, muffled sound, warm tape hiss"
]

prompts1 = [
    "Upbeat synth-pop music",
    "Solo classical piano sonata",
    "Epic full orchestral symphony",
    "Fast bebop jazz improvisation",
    "Heavy metal electric guitar riff",
    "Deep house electronic dance music",
    "Acoustic folk singer songwriter",
    "Reggae band with heavy bass",
    "Old school boom-bap hip hop",
    "Country music with banjo and fiddle",
    "Experimental avant-garde jazz",
    "Hard rock drum solo",
    "Ambient chillout synthesizer pads",
    "Traditional blues harmonica",
    "High-energy disco funk",
    "Gothic church pipe organ music",
    "Latin salsa with brass section",
    "Punk rock with distorted vocals",
    "Soul music with rhythmic piano",
    "Chiptune 8-bit video game music"
]


output_dir = "audio_outputs_no_sparke"
os.makedirs(output_dir, exist_ok=True)

# 6. GENERATE AND SAVE AUDIO
for i, prompt in enumerate(prompts, start=1):
    print(f"Generating audio {i}: {prompt}")
    generator = torch.Generator(device=device).manual_seed(i)
    
    output = pipe(
        prompt=prompt,
        generator=generator
    )

    audio = output.audios[0]
    
    # CRITICAL FIX: Convert float32 [-1.0, 1.0] to 16-bit PCM for standard media players
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)
    
    output_path = os.path.join(output_dir, f"baseline_audio{i}.wav")
    scipy.io.wavfile.write(output_path, rate=16000, data=audio_int16) # Save the int16 version
    print(f"Saved {output_path}")

print("All audio files generated successfully!")


#output_dir = "audio_outputs_no_sparke"
#output_path = os.path.join(output_dir, f"baseline_audio{i}.wav")