import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import laion_clap
import soundfile as sf
import librosa
from scipy.linalg import sqrtm
from panns_inference import AudioTagging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Configuration & Directories
# ==============================================================================
REF_JAZZ_DIR = r"C:\Users\sepan\sparke-photo to audio\reference-jazz"

dir_list = [
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_cond-vendi-FKEA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_cond-vendi-FKEA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs1_cond-vendi-FKEA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs2_cond-vendi-FKEA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs3_cond-vendi-FKEA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_cond-vendi-CSA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs1_cond-vendi-CSA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs2_cond-vendi-CSA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs3_cond-vendi-CSA",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_cond-vendi",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs1_cond-vendi",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs2_cond-vendi",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs3_cond-vendi",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_cond-rke",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs1_cond-rke",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs2_cond-rke",
    r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs3_cond-rke",
]

dir_list =[r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_best_cond-vendi-FKEA",
           r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_best_cond-vendi-CSA",
            r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_best_cond-vendi",
            r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_best_cond-rke"
]
dir_list = [
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke4_new_cond-rke",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke4_new_cond-vendi",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke4_new_cond-vendi-CSA",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke4_new_cond-vendi-FKEA",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke5_new_cond-rke",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke5_new_cond-vendi",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke5_new_cond-vendi-CSA",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke5_new_cond-vendi-FKEA",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke6_new_cond-rke",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke6_new_cond-vendi",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke6_new_cond-vendi-CSA",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke6_new_cond-vendi-FKEA",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke_cond-rke",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke_cond-vendi",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke_CSA",
    r"C:\Users\sepan\sparke-photo to audio\tangos\tango_sounds_sparke_FKEA",
]
dir_list= [r"C:\Users\sepan\sparke-photo to audio\sparke_audioldm2_audio_outputs_general"]

chosen_numbers = list(range(1, 80))

# ==============================================================================
# 2. Pre-Validation & Distance Helpers
# ==============================================================================
def filter_valid_audio_files(file_list):
    """Filters out corrupted, empty, or unreadable audio files."""
    valid_files = []
    for f in file_list:
        try:
            info = sf.info(f)
            if info.duration > 0:
                valid_files.append(f)
            else:
                print(f"[Warning] Skipped zero-duration file: {os.path.basename(f)}")
        except Exception as e:
            print(f"[Warning] Skipped corrupt file '{os.path.basename(f)}': {e}")
    return valid_files


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculates 2-Wasserstein / Fréchet Distance between two feature distributions."""
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1_cov = sigma1 + offset
    sigma2_cov = sigma2 + offset

    covmean, _ = sqrtm(sigma1_cov.dot(sigma2_cov), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def compute_clap_fad_from_stats(mu_ref, sigma_ref, gen_paths, clap_model):
    """Computes FAD using pre-computed reference stats and generated audio embeddings."""
    with torch.no_grad():
        gen_embeds = clap_model.get_audio_embedding_from_filelist(x=gen_paths, use_tensor=False)

    mu_gen, sigma_gen = np.mean(gen_embeds, axis=0), np.cov(gen_embeds, rowvar=False)
    return calculate_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)


def compute_panns_pairwise_diversity(gen_paths, panns_model):
    """Compute the mean pairwise cosine distance between PANNs CNN14 embeddings using pre-loaded model."""
    if len(gen_paths) < 2:
        print("[Error] At least two valid generated audio files are required.")
        return None

    embeddings = []
    for idx, audio_path in enumerate(gen_paths, start=1):
        try:
            audio, _ = librosa.load(audio_path, sr=32000, mono=True)
            if audio.size == 0:
                print(f"[Warning] Empty audio skipped: {os.path.basename(audio_path)}")
                continue

            audio = audio.astype(np.float32)[None, :]
            _, embedding = panns_model.inference(audio)
            embeddings.append(np.asarray(embedding, dtype=np.float32).squeeze())
            print(f"\rExtracted PANNs embeddings: {idx}/{len(gen_paths)}", end="", flush=True)
        except Exception as e:
            print(f"\n[Warning] PANNs embedding failed for '{os.path.basename(audio_path)}': {e}")

    print()
    if len(embeddings) < 2:
        print("[Error] Fewer than two valid PANNs embeddings were extracted.")
        return None

    embeddings = np.stack(embeddings, axis=0)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    cosine_similarity = embeddings @ embeddings.T
    cosine_distance = 1.0 - cosine_similarity
    upper_triangle = np.triu_indices(len(embeddings), k=1)
    pairwise_distances = cosine_distance[upper_triangle]

    mean_distance = float(np.mean(pairwise_distances))
    std_distance = float(np.std(pairwise_distances, ddof=1)) if len(pairwise_distances) > 1 else 0.0
    return mean_distance, std_distance, len(embeddings)


# ==============================================================================
# 3. Prompt List (80 Jazz Prompts)
# ==============================================================================
prompts = [
    # 1 - 20
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

    # 21 - 40
    "Cool jazz, muted Miles Davis style trumpet, relaxed rhythm section, late night mood",
    "Hard bop, aggressive tenor sax, driving piano rhythm, high energy",
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
    "Post-bop jazz, modal chord progressions, intricate polyrhythmic drumming, adventurous solos",

    # 41 - 80
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

# ==============================================================================
# 4. Main Execution Pipeline Across All Directories
# ==============================================================================
if __name__ == "__main__":
    # --- Step A: Load Models Once ---
    print(f"\nLoading CLAP model on device: {device}...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt(model_id=1)
    clap_model.to(device)

    print(f"Loading PANNs AudioTagging model...")
    panns_device = "cuda" if device.type == "cuda" else "cpu"
    panns_model = AudioTagging(checkpoint_path=None, device=panns_device)

    # --- Step B: Precompute Reference Embeddings for FAD Once ---
    print("\nPre-computing Reference Jazz Audio Statistics for CLAP-FAD...")
    raw_ref_files = sorted(
        glob.glob(os.path.join(REF_JAZZ_DIR, "*.wav")) + glob.glob(os.path.join(REF_JAZZ_DIR, "*.mp3"))
    )
    ref_files = filter_valid_audio_files(raw_ref_files)

    if len(ref_files) >= 10:
        print(f"Extracting CLAP embeddings for {len(ref_files)} reference audio files...")
        with torch.no_grad():
            ref_embeds = clap_model.get_audio_embedding_from_filelist(x=ref_files, use_tensor=False)
        mu_ref, sigma_ref = np.mean(ref_embeds, axis=0), np.cov(ref_embeds, rowvar=False)
    else:
        print(f"[Warning] Reference directory '{REF_JAZZ_DIR}' needs at least 10 valid files. FAD skipped.")
        mu_ref, sigma_ref = None, None

    results_summary = []

    # --- Step C: Loop Through Each Directory in `dir_list` ---
    for dir_idx, gen_dir in enumerate(dir_list, start=1):
        folder_name = os.path.basename(gen_dir)
        print("\n" + "="*70)
        print(f" [{dir_idx}/{len(dir_list)}] EVALUATING: {folder_name}")
        print(f" Path: {gen_dir}")
        print("="*70)

        if not os.path.exists(gen_dir):
            print(f"[Error] Directory does not exist: {gen_dir}")
            continue

        raw_audio_paths = []
        raw_prompts = []
        for k in chosen_numbers:
            file_path = os.path.join(gen_dir, f"tango_jazz_{k}.wav")
            if os.path.exists(file_path):
                raw_audio_paths.append(file_path)
                raw_prompts.append(prompts[k - 1])

        all_audio_paths = []
        chosen_prompts = []
        for path, prompt in zip(raw_audio_paths, raw_prompts):
            if len(filter_valid_audio_files([path])) > 0:
                all_audio_paths.append(path)
                chosen_prompts.append(prompt)

        print(f"Matched {len(all_audio_paths)} valid generated audio files.")
        if len(all_audio_paths) == 0:
            print(f"[Error] No valid audio files found in directory.")
            continue

        # 1. Compute CLAP Score (Text-Audio Alignment)
        print("1/3 Computing CLAP Score...")
        with torch.no_grad():
            audio_embeds = clap_model.get_audio_embedding_from_filelist(x=all_audio_paths, use_tensor=True)
            if not isinstance(audio_embeds, torch.Tensor):
                audio_embeds = torch.from_numpy(audio_embeds)
            audio_embeds = audio_embeds.to(device).float()

            text_embeds = clap_model.get_text_embedding(chosen_prompts)
            if not isinstance(text_embeds, torch.Tensor):
                text_embeds = torch.from_numpy(text_embeds)
            text_embeds = text_embeds.to(device).float()

            audio_norm = F.normalize(audio_embeds, dim=-1)
            text_norm = F.normalize(text_embeds, dim=-1)
            clap_score = (audio_norm * text_norm).sum(dim=-1).mean().item()

        # 2. Compute CLAP-FAD Score
        print("2/3 Computing CLAP-FAD...")
        if mu_ref is not None and sigma_ref is not None:
            fad_score = compute_clap_fad_from_stats(
                mu_ref=mu_ref,
                sigma_ref=sigma_ref,
                gen_paths=all_audio_paths,
                clap_model=clap_model
            )
        else:
            fad_score = None

        # 3. Compute PANNs Pairwise Diversity Score
        print("3/3 Computing PANNs Pairwise Diversity...")
        panns_result = compute_panns_pairwise_diversity(
            gen_paths=all_audio_paths,
            panns_model=panns_model
        )

        panns_mean = panns_result[0] if panns_result is not None else None
        panns_std = panns_result[1] if panns_result is not None else None

        # Print per-folder summary
        print(f"\n--- Results for {folder_name} ---")
        print(f" CLAP Score      : {clap_score:.4f}")
        print(f" CLAP-FAD        : {fad_score:.4f}" if fad_score is not None else " CLAP-FAD        : FAILED")
        print(f" PANNs Diversity : {panns_mean:.4f} ± {panns_std:.4f}" if panns_mean is not None else " PANNs Diversity : FAILED")

        results_summary.append({
            "folder": folder_name,
            "count": len(all_audio_paths),
            "clap": clap_score,
            "fad": fad_score,
            "panns_mean": panns_mean,
            "panns_std": panns_std
        })

    # ==============================================================================
    # 5. Final Grand Summary Table
    # ==============================================================================
    print("\n" + "="*85)
    print("                             FINAL EVALUATION SUMMARY")
    print("="*85)
    header = f"{'Folder Name':<50} | {'Count':<5} | {'CLAP ↑':<8} | {'CLAP-FAD ↓':<10} | {'PANNs Div ↑':<15}"
    print(header)
    print("-" * len(header))

    for res in results_summary:
        folder_str = res['folder'][:50]
        cnt_str = str(res['count'])
        clap_str = f"{res['clap']:.4f}" if res['clap'] is not None else "N/A"
        fad_str = f"{res['fad']:.4f}" if res['fad'] is not None else "N/A"
        panns_str = f"{res['panns_mean']:.4f}±{res['panns_std']:.3f}" if res['panns_mean'] is not None else "N/A"

        print(f"{folder_str:<50} | {cnt_str:<5} | {clap_str:<8} | {fad_str:<10} | {panns_str:<15}")

    print("="*85 + "\n")