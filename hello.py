import torch
import soundfile as sf
import os
import sys

# Add tango-master to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tango-master', 'tango2'))

try:
    from tango import Tango
except ImportError:
    print("Using fallback Tango implementation...")
    from tango import Tango

def main():
    # 1. Start the engine
    print("(Guitar) Waking up Tango and loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        model = Tango("declare-lab/tango2", device=device)


# --- SPEED OPTIMIZATIONS FOR RTX 4070 ---
        print("🚀 Optimizing for float16 speed...")
        model.vae.to(dtype=torch.float16)
        
        # In Tango's custom class, the core diffusion engine is usually model.model
        if hasattr(model, 'model'):
            model.model.to(dtype=torch.float16)
            
        # Tango's text encoder (T5) is often tucked inside model.model or called 'text_encoder' 
        # but since 'text_encoder' failed, let's target the components safely:
        if hasattr(model, 'text_encoder'):
             model.text_encoder.to(dtype=torch.float16)
        
        # Some versions of Tango use 'stft' for the audio processing
        if hasattr(model, 'stft'):
            model.stft.to(dtype=torch.float16)
        # ----------------------------------------


    except Exception as e:
        import traceback
        print(f"Error loading tango model: {e}")
        traceback.print_exc()
        return

    # 2. Your List of 5 Prompts
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
    # 3. Create the output folder
    output_folder = "tango_sounds"
    os.makedirs(output_folder, exist_ok=True)
    print(f"(Music) Starting generation for {len(prompts)} tracks in '{output_folder}'...")
    # 4. Generate tracks
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            # i+1 / len(prompts) ensures the counter shows [1/20], [2/20], etc.
            print(f"🎵 Processing [{i+1}/{len(prompts)}]: {prompt}")
            
            try:
                # steps=30 for the best speed/quality balance on your 4070
                audio = model.generate_for_batch([prompt], samples=1, steps=30)[0]

                # 5. Save each track with a clean filename inside tango_sounds
                filename = os.path.join(output_folder, f"tango_jazz_{i+1}.wav")
                sf.write(filename, audio, samplerate=16000)
                print(f"[OK] Saved: {filename}")
                
            except Exception as e:
                print(f"[SKIP] Could not generate track {i+1}: {e}")
if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] All tracks finished! Check the 'jazz_outputs' folder.")
    except Exception as e:
        print(f"[ERROR] Something went wrong: {e}")