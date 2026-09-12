import torch
import torch.nn.functional as F
import laion_clap
import glob
import librosa
import numpy as np
import csv
import os
import json
from conditional_evaluation import ConditionalEvaluation 

# --- NEW IMPORT FOR FAD ---
from frechet_audio_distance import FrechetAudioDistance

os.environ["USE_QWEN_QUALITY_CHECK"] = "true"

# 1. Load the pre-trained CLAP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_QWEN_QUALITY_CHECK = os.getenv("USE_QWEN_QUALITY_CHECK", "false").lower() in {"1", "true", "yes", "on"}
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


def assess_audio_quality(audio_file):
    audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)
    if audio_data.size == 0 or not np.isfinite(audio_data).all():
        return {"file": audio_file, "score": 0.0, "good": False, "reason": "empty_or_invalid"}

    peak = float(np.max(np.abs(audio_data)))
    clipping_fraction = float(np.mean(np.abs(audio_data) >= 0.999))
    rms = librosa.feature.rms(y=audio_data)[0]
    rms_db = float(librosa.amplitude_to_db(np.maximum(np.mean(rms), 1e-10), ref=1.0))
    silence_threshold = max(float(np.max(rms)) * 0.02, 1e-5)
    silence_fraction = float(np.mean(rms <= silence_threshold))
    duration_seconds = float(audio_data.size / sample_rate)

    score = 100.0
    if duration_seconds < 3.0:
        score -= 30.0
    if rms_db < -40.0:
        score -= 25.0
    elif rms_db > -1.0:
        score -= 15.0
    score -= min(clipping_fraction * 5000.0, 40.0)
    score -= min(max(silence_fraction - 0.35, 0.0) * 70.0, 35.0)
    score = max(0.0, min(score, 100.0))

    return {
        "file": audio_file,
        "score": round(score, 2),
        "good": score >= 60.0,
        "duration_seconds": round(duration_seconds, 2),
        "rms_db": round(rms_db, 2),
        "peak": round(peak, 5),
        "clipping_fraction": round(clipping_fraction, 6),
        "silence_fraction": round(silence_fraction, 4),
    }


def assess_audio_quality_with_qwen(audio_file, prompt, processor, model):
    sample_rate = processor.feature_extractor.sampling_rate
    audio_data, _ = librosa.load(audio_file, sr=sample_rate, mono=True)
    instruction = f"""
Listen to this generated audio file and evaluate its quality.
Requested description: {prompt}

Return only JSON with these fields:
{{
  "overall_score": number from 0 to 10,
  "prompt_match_score": number from 0 to 10,
  "has_obvious_artifacts": true or false,
  "good": true or false,
  "reason": "brief explanation"
}}

Mark good true only when the audio is pleasant and coherent, has no obvious
generation artifacts, and reasonably matches the requested description.
"""
    conversation = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": audio_file},
            {"type": "text", "text": instruction},
        ],
    }]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audio=[audio_data],
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_text = processor.batch_decode(
        generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0]
    try:
        json_start = generated_text.find("{")
        json_end = generated_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON block found in output.")
        
        result = json.loads(generated_text[json_start:json_end])
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse Qwen JSON for {audio_file}. Raw output: {generated_text}")
        result = {"overall_score": 0, "prompt_match_score": 0, "has_obvious_artifacts": True, "good": False, "reason": "JSON Parsing Error"}
    return {
        "qwen_overall_score": result.get("overall_score"),
        "qwen_prompt_match_score": result.get("prompt_match_score"),
        "qwen_has_obvious_artifacts": result.get("has_obvious_artifacts"),
        "qwen_good": result.get("good"),
        "qwen_reason": result.get("reason", ""),
    }


def write_quality_report(audio_files, output_directory, prompts=None):
    quality_results = [assess_audio_quality(audio_file) for audio_file in audio_files]

    if not USE_QWEN_QUALITY_CHECK:
        print("Skipping Qwen2-Audio quality check (opt-in via USE_QWEN_QUALITY_CHECK=true).")
        return quality_results

    try:
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        qwen_model_name = os.getenv("QWEN_AUDIO_MODEL", "Qwen/Qwen2-Audio-7B-Instruct")
        qwen_processor = AutoProcessor.from_pretrained(qwen_model_name)
        qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            qwen_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        for index, result in enumerate(quality_results):
            if not result["good"]:
                continue
            prompt = prompts[index] if prompts and index < len(prompts) else "Generated audio"
            result.update(assess_audio_quality_with_qwen(
                result["file"], prompt, qwen_processor, qwen_model
            ))
            result["good"] = bool(result["qwen_good"])
    except Exception as error:
        print(f"Qwen2-Audio quality check failed: {error}")

    quality_results.sort(key=lambda result: result["score"], reverse=True)
    good_audio_files = [result["file"] for result in quality_results if result["good"]]

    report_path = os.path.join(output_directory, "audio_quality_report.csv")
    good_files_path = os.path.join(output_directory, "good_quality_audio_files.txt")
    field_names = sorted({key for result in quality_results for key in result})
    with open(report_path, "w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(report_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(quality_results)
    with open(good_files_path, "w", encoding="utf-8") as good_files_file:
        good_files_file.write("\n".join(good_audio_files))

    print(f"Good-quality audio files: {len(good_audio_files)}/{len(audio_files)}")
    print(f"Detailed quality report: {report_path}")
    print(f"Good audio file list: {good_files_path}")
    return quality_results

# 2. Collect Audio Files
all_audio_paths = []
chosen_numbers = [
    1, 2, 3, 4, 5, 7, 8, 9, 10, 14,
    16, 18, 19, 20, 21, 22, 23, 24, 26, 27,
    28, 29, 30, 33, 34, 35, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 57, 59, 60, 61, 62,
    64, 65, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79
]

#chosen_numbers = list(range(1, 80))  # Adjust this range based on the number of audio files you have
for k in chosen_numbers:
    #pattern = f"C:\\Users\\sepan\\sparke-photo to audio\\sparke_audioldm2_audio_outputs_general\\sparke_audio_{k}.wav"
    pattern = f"C:\\Users\\sepan\\sparke-photo to audio\\audio_outputs_no_sparke\\sparke_audio_{k}.wav"
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

    ##########################################################
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
    "Late-night radio ballad, lush muted horn section, gentle brushed snare, romantic atmosphere"
]

chosen_prompts = []
  # Ensure the number of prompts matches the number of audio files
for k in chosen_numbers:
    chosen_prompts.append(prompts1[k-1])
T_features = extract_t_features(model, chosen_prompts, device)

# 4. Evaluation
evaluator = ConditionalEvaluation(sigma=0.5)
results = evaluator.conditional_entropy(X=X_features, Y=T_features, order=1)

# Results extraction
cond_entropy = results[0] 
cond_vendi_score = torch.exp(cond_entropy)

print(f"--- Results ---")
print(f"Conditional-Vendi Score after applying SPARKE(on the jazz dataset): {cond_vendi_score:.4f}")

print("All audio files generated successfully!")

