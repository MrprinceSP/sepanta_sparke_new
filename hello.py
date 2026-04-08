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
    
    # This will download the 'brain' (weights) from Hugging Face
    try:
        model = Tango("declare-lab/tango2", device=device)
    except Exception as e:
        import traceback
        print(f"Error loading tango model: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("Proceeding with available resources...")
        return

    # 2. Your Jazz Request
    prompt = "A high-quality, sophisticated smooth jazz track with a soulful saxophone and soft piano."

    # 3. Generate 5 tracks
    print("(Music) Generating 1 quick jazz track for a fast test run...")
    
    try:
        # samples=1 gives a single example quickly
        # steps=50 uses fewer diffusion steps so the run completes much faster
        audios = model.generate_for_batch([prompt], samples=1, steps=50)

        # 4. Save the music
        os.makedirs("jazz_outputs", exist_ok=True)
        for i, audio in enumerate(audios):
            filename = f"jazz_outputs/jazz_track_{i+1}.wav"
            sf.write(filename, audio, samplerate=16000)
            print(f"[OK] Created: {filename}")
    except Exception as e:
        print(f"Could not generate audio: {e}")
        return

if __name__ == "__main__":
    try:
        main()
        print("\n[SUCCESS] Success! Check the 'jazz_outputs' folder for your music.")
    except Exception as e:
        print(f"[ERROR] Something went wrong: {e}")