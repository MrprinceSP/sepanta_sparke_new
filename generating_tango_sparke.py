import torch
import soundfile as sf
import os
import sys

sys.path.insert(0, "src")
import time
import numpy as np # <--- Added numpy for audio saving
import logging
import scipy.io.wavfile 
print("defefwef")
# Add custom path to Tango if needed
try:
    from tango import Tango
except ImportError:
    print("Using fallback Tango implementation...")
    from src.tango import Tango  # Or your actual fallback path

# 1. IMPORT YOUR MODIFIED PIPELINE AND RKE UTILS
# (Ensure these paths match your local file structure)

from diffusers.pipelines.tango.pipeline_tango_sparke import TangoPipeline
from diffusers.pipelines.rke_guidance_utils_tango import RKEGuidedSampling

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

start_time = time.perf_counter()
print("Initializing Tango with SPARKE Guidance...")

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tango utilizes FLAN-T5, which can behave unpredictably with fp16 in some environments. 
# Depending on your GPU, torch.float32 is the safest bet to avoid NaNs.
torch_dtype = torch.float32 

# 2. LOAD THE TANGO PIPELINE COMPONENTS MANUALLY
repo_id = "declare-lab/tango2" 
logger.info(f"Downloading/Loading official {repo_id} weights...")

# 2a. Use the official Tango loader to fetch the raw components
tango_raw = Tango(repo_id, device=device)

# 2b. Extract the loaded models
vae = tango_raw.vae
stft = tango_raw.stft
unet = tango_raw.model.unet
text_encoder = tango_raw.model.text_encoder
tokenizer = tango_raw.model.tokenizer
scheduler = tango_raw.scheduler

logger.info("Initializing custom SPARKE Diffusers Pipeline...")
vae.vocoder.to(device)

# 2c. Manually initialize your custom Diffusers pipeline
# (Note: If your custom TangoPipeline __init__ uses slightly different 
# variable names, adjust them below to match)

pipe = TangoPipeline(
    vae=vae,
    vocoder=vae.vocoder,
    unet=unet,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler
)
# 3. SET UP RKE GUIDED SAMPLING
# Note: Since we implemented the prompt_embeds fix inside RKEGuidedSampling, 
# use_latents_for_guidance MUST be True for it to utilize FLAN-T5 embeddings and audio latents.
rke_guided_sampler = RKEGuidedSampling(
    algorithm="cond-rke-FKEA", 
    kernel="gaussian",
    sigma_image=0.6,     # Operates on the audio latents
    sigma_text=0.3,      # Operates on the pooled FLAN-T5 embeddings
    max_bank_size=1000,
    use_latents_for_guidance=True, 
    model_name="tango",  # Identifies that we are not using standard SD/AudioLDM
    model=pipe
)

# 4. GUIDANCE PARAMETERS
# Tango natively doesn't have `clap_for_guidance` since it uses FLAN-T5. 
# If your modified pipeline relies on `vscore_clap`, ensure you either pass a separately loaded CLAP model 
# here, or change the criteria if your code doesn't strictly require it.
guidance_kwargs = dict(
    num_inference_steps=80,      # 200 is recommended by Declare-Lab for Tango/Tango2
    guidance_scale=7.5,           # 3.0 is the standard CFG scale for Tango models
    num_waveforms_per_prompt=1,   # Number of audios to generate per prompt
    
    # SPARKE Specific Parameters
    rke_guided_sampler=rke_guided_sampler,
    criteria_guidance_scale=0.05, 
    
    # IMPORTANT: If your criteria is "vscore_clap", you need a CLAP model. 
    # If your pipeline doesn't automatically load one, you may need to initialize it.
    # Otherwise, you can leave it as None if your modified code relies purely on FLAN-T5 prompt_embeds.
    clap_for_guidance=None,       
    
    guidance_freq=1,
    criteria="vscore",            # Changed from vscore_clap to general vscore to avoid missing CLAP errors
    F_M=None,
    F_T=None,
    F_M_real=None,
    F_T_real=None, 
    beta=0,
    regularize=False,
    regularize_weight=0,
)
    # 2. Your List of 5 Prompts
# 5. PROMPTS TO GENERATE
prompts = [
    # --- Original 20 Prompts ---
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
    "Low-fidelity bedroom jazz, muffled sound, warm tape hiss",

    # --- 20 New Prompts to reach 40 ---
    "Cool jazz, muted Miles Davis style trumpet, relaxed rhythm section, late night mood",
    "Hard bop, aggressive tenor sax, driving driving piano rhythm, high energy",
    "Latin jazz, heavy conga drumming, bright piano montuno, energetic brass",
    "Acid jazz, electronic synth bassline, jazzy Rhodes electric piano, hip-hop backbeat",
    "Ethio-jazz, hypnotic pentatonic scales, vintage farfisa organ, brass stabs",
    "Chamber jazz, classical strings mixed with improvisational jazz flute and oboe",
    "Spiritual jazz, meditative saxophone drones, shimmering chimes, cosmic atmosphere",
    "Dixieland jazz, polyphonic collective improvisation, clarinet, trombone, banjo",
    "Gypsy swing jazz ensemble, dual acoustic guitars, upright bass, violin solo",
    "Cape Town jazz, bright marimba rhythms, uplifting horn arrangements, coastal energy",
    "Dark jazz fusion, distorted electric bass, industrial drums, haunting sax wails",
    "Minimalist jazz piano, sparse arrangement, plenty of silence between notes, warm room tone",
    "Solo double bass performance, plucking and bowing techniques, deep resonant low-end",
    "Vintage jazz quartet, mono radio broadcast quality, 1940s sound texture",
    "Outdoor festival jazz recording, faint wind blowing, cheering crowd in background",
    "Jazz fusion, slap bass solo, complex time signature, synthesized keyboards",
    "Slow tempo blues-jazz ballad, expressive electric guitar bends, late-night bar vibe",
    "Orchestral jazz, sweeping cinematic strings accompanying a solo grand piano",
    "Lo-fi hip-hop jazz, dusty vinyl crackle, sampled jazz guitar loops, relaxed boom-bap beat",
    "Post-bop jazz, modal chord progressions, intricate polyrhythmic drumming, adventurous solos"
    ##############################################################################another 40
    "Japanese 80s city pop jazz fusion, slap bass, smooth synthesizers, electric guitar solos",
    "Nu-jazz with broken beat rhythms, electronic synth textures, and live saxophone",
    "Acid-funk jazz with clavinet, talkbox effects, and tight pocket drumming",
    "Psychedelic jazz-rock with electric violin, distorted organ, and soaring guitar",
    "ECM-style Nordic jazz, atmospheric trumpet with long digital delays, pristine spatial reverb",
    "Jazz-hop beat with sampled Rhodes piano, crisp hi-hats, and warm acoustic bass",
    "Dark ambient post-jazz with bowed cymbals, pitch-shifted sax wails, and slow drone",
    "Modern M-Base jazz with complex overlapping time signatures and sharp alto sax",
    "New Orleans second line brass band, heavy tuba bassline, snare roll, energetic street march",
    "Afro-Cuban descarga, improvised mambo brass, driving congas, timbales solo",
    "Brazilian samba-jazz, fast acoustic nylon guitar, lively pandeiro, soaring flute",
    "Parisian hot club swing, gypsy rhythm guitar, virtuoso accordion, wooden double bass",
    "Argentine tango-jazz, dramatic bandoneon, staccato piano, expressive violin",
    "Japanese spiritual jazz, bamboo shakuhachi flute, harp, freeform percussion",
    "Kansas City jump blues jazz, honking tenor sax, boogie-woogie piano, swinging rhythm",
    "South American bossa-jazz waltz in 3/4 time, gentle vocal hums, soft nylon guitar",
    "1910s ragtime piano, stride left hand, syncopated right hand, upright piano sound",
    "1930s big band swing contest, roaring brass riffs, driving four-on-the-floor kick, energetic pace",
    "West Coast cool jazz, French horn, muted trombone, gentle relaxed tempo",
    "Third Stream jazz, chamber woodwinds mixed with improvisational jazz rhythm section",
    "Gospel-infused soul jazz, warm Fender Rhodes, swelling Hammond B3 organ, bluesy phrasing",
    "Modal jazz, static two-chord modal vamp, expansive soprano saxophone exploration",
    "Classic organ trio, Hammond B3 drawbar organ, hollow-body electric guitar, drum brushes",
    "Early Dixieland brass, polyphonic clarinet, washboard percussion, banjo strumming",
    "Solo jazz harp, ethereal extended chord voicings, cascading arpeggios",
    "Vibraphone-led jazz quartet, crystalline mallet tones, swinging upright bass",
    "Bass clarinet-led chamber jazz ensemble, deep woody bass notes, subtle cello accompaniment",
    "Baritone saxophone lead, heavy low-end riffing, driving barroom jazz groove",
    "Vocal scat and tap dance duet, fast rhythmic footwork, acrobatic vocal improvisation",
    "Solo acoustic guitar, fingerstyle jazz chord melody, rich natural resonance",
    "Flute-led Latin jazz sextet, intricate flute runs, montuno piano, guiro rhythm",
    "Boogie-woogie piano duo, roaring left-hand bass patterns, fast blues riffs",
    "Late-night rain, distant police siren, smoky film noir detective saxophone",
    "Vintage cassette tape demo, saturated audio compression, flutter, warm analog hiss",
    "Sunlit outdoor park performance, ambient birds chirping, gentle breeze, acoustic trio",
    "High-end audiophile studio session, pristine instrument isolation, dynamic punch",
    "Empty auditorium sound check, booming acoustic reflections, solitary trumpet calls",
    "Underground speakeasy, murmur of crowd, ice clinking, unamplified acoustic jazz",
    "1970s radio studio broadcast, tube preamp warmth, dynamic FM radio compression",
    "Late-night radio ballad, lush muted horn section, gentle brushed snare, romantic atmosphere",
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
    "Chiptune 8-bit video game music",
    "Lofi chillhop beats with vinyl crackle",
    "Cinematic sci-fi soundtrack with deep strings",
    "Acoustic bossa nova guitar",
    "Fast-paced bluegrass mandolin picking",
    "Dreamy indie pop with ethereal synthesizers",
    "Traditional Celtic folk music with bagpipes",
    "Intense dubstep drop with heavy wobble bass",
    "Smooth R&B with a slow groove",
    "High-energy K-pop dance track",
    "Minimalist techno with a driving kick drum",
    "90s grunge rock with fuzzy guitars",
    "Traditional Japanese shamisen and taiko drums",
    "Surf rock with spring reverb electric guitar",
    "Slow delta blues slide guitar",
    "Euphoric trance music with uplifting melodies",
    "Afrobeat with complex percussion polyrhythms",
    "Flamenco guitar with rhythmic hand clapping",
    "Modern trap beat with rapid 808 hi-hats",
    "A cappella choral harmony in a grand hall",
    "Psychedelic rock with a swirling Hammond organ"
    ##########################################################################
    "Dark synthwave with 80s analog basslines",
    "Neo-classical cello ensemble with emotional melodies",
    "High-tempo psytrance with driving rolling bass",
    "Motown soul with a tight horn section and rhythmic bass",
    "Math rock with complex time signatures and clean guitars",
    "Progressive metal with heavy djent guitar riffs",
    "Vintage Italian film score with dramatic orchestration",
    "Tropical house with steel drums and a airy flute melody",
    "Traditional Indian sitar and tabla duet",
    "Ska punk with energetic brass and upbeat skank guitar",
    "Atmospheric post-rock with swelling crescendos",
    "Electro swing with 1920s big band brass and electronic beats",
    "Dark ambient drone with metallic textures",
    "Traditional West African kora harp melody",
    "Baroque harpsichord concerto with string orchestra",
    "Shoegaze with dense walls of distorted fuzzy reverb",
    "Modern hyperpop with pitch-shifted vocal chops and synth melodies",
    "Gypsy jazz with fast acoustic swing guitar",
    "Melodic death metal with fast double-kick drums",
    "Drum and bass with fast 174 BPM breakbeats and reese bass",
    "Future bass with bright sidechained synths and pitched vocals",
    "Middle Eastern oud improvisation with frame drum",
    "Post-punk with a driving melodic bassline and minimal guitar",
    "Dramatic operatic soprano aria with full orchestral backing",
    "Industrial techno with pounding distorted kicks",
    "Neo-soul with warm Fender Rhodes electric piano and laid-back groove",
    "Mariachi band with vibrant trumpets and guitarrón",
    "Progressive house with atmospheric chord progressions",
    "Darkwave with melancholic gothic synthesizers and drum machine",
    "Gospel choir with powerful lead vocals and Hammond organ",
    "Folk metal with accordion and heavy guitar riffs",
    "Glitch hop with stuttering electronic beats and heavy funk bass",
    "Romantic era violin concerto with expressive solo performance",
    "Hardstyle electronic with pitch-bent distorted kicks",
    "Hawaiian slack-key acoustic guitar instrumental",
    "Acid jazz with groovy bassline and saxophone solos",
    "Symphonic black metal with blast beats and orchestral elements",
    "Vaporwave with slowed-down 80s funk samples and pitch shifts",
    "Cuban son music with tres guitar and clave rhythm",
    "Speed garage with bouncy 4x4 sub-bassline and vocal stabs"
]


output_dir = "tango_sounds_sparke"
os.makedirs(output_dir, exist_ok=True)

# 6. GENERATE AND SAVE AUDIO
for i, prompt in enumerate(prompts, start=1):
    print(f"Generating audio {i}: {prompt}")
    generator = torch.Generator(device=device).manual_seed(i)
    
    output = pipe(
        prompt=prompt,
        generator=generator,
        **guidance_kwargs
    )

    audio = output.audios[0]
    
    # CRITICAL FIX: Convert float32 [-1.0, 1.0] to 16-bit PCM for standard media players
    audio = np.asarray(audio).squeeze()
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio after squeezing, got shape {audio.shape}")
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)
    
    output_path = os.path.join(output_dir, f"sparke_audio_{i}.wav")
    scipy.io.wavfile.write(output_path, rate=16000, data=audio_int16) # Save the int16 version
    print(f"Saved {output_path}")

print("All audio files generated successfully!")
end_time = time.perf_counter()
print(f"Total time taken: {end_time - start_time:.2f} seconds")
'''

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
        '''
