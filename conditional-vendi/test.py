import torch
import torch.nn.functional as F
import laion_clap
import glob
import librosa
import numpy as np
from conditional_evaluation import ConditionalEvaluation 

# 1. Load the pre-trained CLAP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = laion_clap.CLAP_Module(enable_fusion=True)
model.load_ckpt() 
model.to(device)

def get_audio_embeddings(audio_files):
    embeddings = []
    for file in audio_files:
        audio_data, _ = librosa.load(file, sr=48000)
        audio_data = audio_data.reshape(1, -1)
        with torch.no_grad():
            # Extract and convert to tensor
            audio_embed = model.get_audio_embedding_from_data(x=audio_data)
            embeddings.append(torch.from_numpy(audio_embed))
    return torch.cat(embeddings, dim=0)

def extract_t_features(clap_model, prompt_list, device):
    with torch.no_grad():
        # laion-clap returns a numpy array
        t_features = clap_model.get_text_embedding(prompt_list)
        t_features = torch.from_numpy(t_features).to(device)
        # L2 Normalization is key for Vendi
        t_features = F.normalize(t_features, dim=-1)
    return t_features.float()

# 2. Collect Audio Files
all_audio_paths = []
for i in range(1, 21):
    pattern = f"C:\\Users\\sepan\\sparke-photo to audio\\sparke_audioldm2_audio_outputs_general\\sparke_audio_{i}.wav"
    found_files = glob.glob(pattern)
    if found_files:
        all_audio_paths.append(found_files[0])

# --- ADD THIS CHECK ---
print(f"Total files found: {len(all_audio_paths)}")
if len(all_audio_paths) == 0:
    print("Error: No audio files were found! Check your folder path and naming.")
    # Exit or stop here so you don't get the torch.cat error
    exit()
# ----------------------

# 3. Generate Features
X_features = get_audio_embeddings(all_audio_paths).to(device)

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

T_features = extract_t_features(model, prompts, device)

# 4. Evaluation
evaluator = ConditionalEvaluation(sigma=0.5)
results = evaluator.conditional_entropy(X=X_features, Y=T_features, order=1)

# Results extraction
cond_entropy = results[0] 
cond_vendi_score = torch.exp(cond_entropy)

print(f"--- Results ---")
print(f"Conditional-Vendi Score after applying SPARKE(on the jazz dataset): {cond_vendi_score:.4f}")