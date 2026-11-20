import csv
from datetime import datetime
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

# --- NEW IMPORT FOR CONDITIONAL VENDI SCORE ---
from conditional_evaluation import ConditionalEvaluation 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Configuration & Directories
# ==============================================================================
REF_JAZZ_DIR = r"reference-jazz"


approaches =["cond-vendi-FKEA","cond-vendi-CSA","cond-vendi","cond-rke"]
dir_list= [f"sparke_audioldm2_audio_outputs_big_batch_rke_{i}" for i in range(5)]



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
    # --- 1-80: Original & Adapted Seed Prompts ---
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

    # --- 81-160: Swing, Stride, Ragtime & Early Big Band ---
    "1920s Harlem stride piano, fast syncopation, bright upright piano sound",
    "1930s Savoy Ballroom swing orchestra, roaring brass section, energetic lindy hop tempo",
    "Early ragtime brass band, tuba bassline, syncopated cornet, woodblock percussion",
    "1920s Cotton Club swing, muted trumpet fills, driving acoustic banjo, fast rhythm",
    "Swing revival jazz, slap bass, jumpin' saxophones, pounding floor tom rhythm",
    "1930s big band ballad, soaring alto sax melody, lush trombone harmonies, brushed cymbals",
    "Classic Kansas City swing, boogie piano roll, shouting tenor sax, bouncy rhythm",
    "1920s Chicago style jazz, hot cornet solos, slap clarinet, driving tuba",
    "Big band brass battle, alternating trumpet solos, roaring shout chorus, fast swing",
    "1940s wartime big band, optimistic trumpet fanfare, sweeping sax section, tight rhythm",
    "Early jazz cornet solo, scratchy 78rpm shellac record texture, cup mute tone",
    "Swing era clarinet feature, liquid runs, bouncy rhythm section, warm 1930s tone",
    "Jump blues swing, honking sax, electric guitar rhythm, fast boogie beat",
    "1930s hotel ballroom orchestra, sweet violin section, muted trumpet, gentle swing",
    "Stride piano duel, two pianos, rapid octave bass jumps, virtuosic right hand runs",
    "Big band drum feature, booming kick drum, fast snare press rolls, brass accents",
    "1920s speakeasy jazz, muted trombone slide, acoustic guitar strum, fast tempo",
    "Classic swing guitar rhythm, four-to-the-bar acoustic chord chops, steady double bass",
    "Vintage big band arrangement, flutes and clarinets unison lead, swinging bassline",
    "1930s swing trio, vibraphone lead, acoustic guitar, upright bass, fast tempo",
    "Early jazz washboard band, kazoo solo, strumming banjo, driving tub bass",
    "Swing era vocal group, tight three-part harmony, bouncing brass backing, fast tempo",
    "1940s swing saxophone section, lush five-part harmony, warm analog tape glow",
    "Big band shout chorus, full brass climax, screaming high trumpet notes, driving drums",
    "1920s barrelhouse piano, gritty blues chords, syncopated rhythm, wooden piano frame",
    "Classic Dixieland parade band, booming bass drum, ringing cymbals, polyphonic horns",
    "1930s gypsy swing, hot club rhythm, dual acoustic guitars, fast violin solo",
    "Swing era trombone feature, warm velvet tone, expressive slide vibrato, slow tempo",
    "1940s radio big band theme song, swelling horns, crisp drum rimshots, mono broadcast",
    "Vintage swing trumpet solo, open horn power, blues inflections, fast tempo",
    "Early jazz banjo solo, rapid chord strumming, metallic acoustic resonance, upbeat",
    "1930s dance hall swing, heavy floor-shaking bass drum, swinging saxophone riffs",
    "Big band sax battle, tenor versus alto, rapid trading, high energy crowd",
    "1920s hot jazz quintet, burning cornet lead, expressive clarinet wails, fast tempo",
    "Classic stride blues, deep left-hand bass notes, bluesy right-hand trills",
    "1940s swing ballad, muted trombone lead, gentle piano chords, brushed drum rhythm",
    "Big band brass fanfare, triumphant trumpets, swelling trombones, dramatic tempo",
    "1930s swing violin feature, soaring melodies, rhythmic acoustic guitar backing",
    "Early jazz clarinet solo, chalumeau lower register, warm room acoustic",
    "1920s piano roll recording, mechanical precision, bright syncopated ragtime",
    "Swing era drum solo, tom-tom heavy groove, swinging hi-hat, energetic finale",
    "1940s swing arrangement, flute lead over muted brass, subtle upright bass",
    "Big band rhythm section feature, piano, guitar, bass, and drums in perfect pocket",
    "1930s swing lounge, low lights, muted trumpet lead, slow brushed snare",
    "Early jazz cornet duo, harmonized brass lines, bouncy rhythm guitar",
    "1920s stride piano waltz, 3/4 time signature, elegant left-hand motion",
    "Swing era baritone sax feature, deep resonant low notes, driving rhythm",
    "1940s big band clarinet lead, silky high notes, lush backing saxophones",
    "Big band brass stabs, sharp dynamic accents, driving snare backbeat",
    "1930s hot club swing guitar solo, blazingly fast arpeggios, wooden resonance",
    "Early jazz trombone feature, tailgate style slides, driving tuba rhythm",
    "1920s vintage jazz orchestra, primitive condenser mic recording, warm mono sound",
    "Swing era vibraphone solo, rapid mallet runs, lush motor tremolo, medium tempo",
    "1940s swing blues, slow 12-bar progression, expressive sax solos, heavy bass",
    "Big band sax section ballad, warm velvet blend, delicate cymbal brushes",
    "1930s swing dance contest, frantic tempo, screaming brass, energetic rhythm section",
    "Early jazz trumpet solo, plunger mute wah-wah effects, low-down blues feel",
    "1920s stride piano stomp, heavy syncopated bass, driving tempo, wooden floor vibration",
    "Swing era guitar chord melody, smooth voice leading, warm archtop guitar tone",
    "1940s big band outro, dramatic brass swell, final cymbal crash, decay reverb",
    "Big band trumpet section, precise high-register unison, crisp rimshots",
    "1930s swing clarinet and sax duet, intertwining lines, buoyant rhythm",
    "Early jazz acoustic bass feature, heavy gut string plucks, steady tempo",
    "1920s ragtime ensemble, syncopated flute, banjo, upright bass, fast tempo",
    "Swing era vocal ballad, torch song, lush brass backing, slow tempo",
    "1940s big band swing march, driving snare drum, uplifting brass melody",
    "Big band trombone section, warm harmonized swells, gentle brushed drums",
    "1930s swing piano solo, light touch, flying right-hand runs, solid left hand",
    "Early jazz trio, cornet, piano, washboard, energetic street corner vibe",
    "1920s stride piano blues, slow tempo, deep low-end rumble, expressive trills",
    "Swing era alto sax solo, warm tone, scoop technique, relaxed medium swing",
    "1940s big band jump swing, call-and-response brass, heavy bass drum",
    "Big band sax section feature, fast unison bebop lines ahead of its time",
    "1930s swing accordion solo, French bistro style, acoustic guitar rhythm",
    "Early jazz trumpet and clarinet duel, fast counterpoint, high energy",
    "1920s stride piano feature, virtuosic speed, syncopated accents, mono clarity",
    "Swing era double bass solo, woody plucks, subtle drum brush accompaniment",
    "1940s big band finale, massive brass wall of sound, thunderous drum roll",
    "Big band muted brass feature, harmon mutes, tight percussive playing",
    "1930s swing jam session, relaxed informal vibe, extended solos, warm room acoustic",

    # --- 161-240: Bebop, Hard Bop & Post-Bop ---
    "1940s bebop quintet, blazing fast alto sax, trumpet unison head, rapid bass plucks",
    "1950s hard bop, soulful tenor sax lead, driving gospel-influenced piano, heavy hi-hat",
    "Post-bop quartet, complex modal harmonies, polyrhythmic drum fills, adventurous solos",
    "Fast bebop piano solo, lightning right-hand runs, sharp left-hand comping",
    "Hard bop trumpet feature, fiery high-note runs, blues-inflected phrasing, driving tempo",
    "1950s bebop jam session, fast tempo, extended tenor saxophone improvisation",
    "Post-bop modal exploration, soprano sax lead, shifting drum patterns, deep bass drone",
    "Bebop drum solo, rapid snare drops, intricate ride cymbal technique, high speed",
    "Hard bop organ quartet, burning Hammond B3 organ, soulful sax, punchy drums",
    "Post-bop piano trio, angular chord voicings, interactive bass and drums, medium tempo",
    "1940s bebop trumpet solo, dizzying chromatic runs, bright brass tone",
    "Hard bop blues, 12-bar form, gritty saxophone wails, heavy swinging bass",
    "Post-bop sax and trumpet unison, intricate fast melody, unexpected accents",
    "Bebop guitar solo, rapid linear lines, clean hollow-body tone, fast tempo",
    "Hard bop quintet, catchy brass riff head, driving tempo, energetic drum response",
    "Post-bop metric modulation, shifting time feel, expressive bass solo",
    "1950s bebop bass solo, fast walking bassline, precise acoustic gut string tone",
    "Hard bop tenor sax solo, deep smoky tone, bluesy bends, high energy",
    "Post-bop ballad, complex reharmonization, subtle cymbal work, expressive trumpet",
    "Bebop trombone feature, surprisingly fast slide work, warm brass tone",
    "Hard bop piano solo, percussive block chords, bluesy runs, driving rhythm",
    "Post-bop quartet, sparse piano comping, searching sax solos, dynamic drum surges",
    "1940s bebop recording, dynamic live club energy, clinking glasses, fast tempo",
    "Hard bop brass duel, trumpet and tenor sax trading fours, intense energy",
    "Post-bop modal waltz, 3/4 time signature, expansive piano chords, rolling drums",
    "Bebop alto sax solo, Charlie Parker style, rapid blues phrasing, bright tone",
    "Hard bop drum feature, press rolls, sharp rimshots, driving ride cymbal pace",
    "Post-bop trumpet feature, open horn, wide interval jumps, atmospheric space",
    "1950s hard bop club date, warm tube recording, heavy bass foundation",
    "Bebop piano and guitar unison lead, tight synchronization, fast tempo",
    "Hard bop gospel-jazz track, churchy piano chords, soulful sax lead, handclaps",
    "Post-bop piano feature, lush altered chords, wide dynamic range, room reverb",
    "Bebop vibraphone solo, rapid mallet technique, clear ringing tones, fast tempo",
    "Hard bop quintet ballad, expressive muted trumpet, gentle piano, brushed snare",
    "Post-bop saxophone solo, sheets of sound, rapid arpeggios, intense energy",
    "1940s bebop double bass feature, rapid walking line, steady pulse, warm wood tone",
    "Hard bop soul-blues, greasy organ riff, honking sax, tight drum pocket",
    "Post-bop rhythmic counterpoint, independent drum and bass patterns, sharp trumpet",
    "Bebop trumpet and sax trading twos, rapid back-and-forth solos, fast tempo",
    "Hard bop piano trio, swinging uptempo groove, virtuosic bass fills",
    "Post-bop modal vamp, repetitive bass line, expansive tenor sax exploration",
    "1950s hard bop anthem, iconic horn intro, driving swing beat, energetic room",
    "Bebop rhythm section, rock-solid walking bass, feathering kick drum, crisp ride",
    "Hard bop alto sax solo, sharp attack, bluesy inflections, high-speed swing",
    "Post-bop trumpet and flute unison, ethereal yet complex harmony, medium tempo",
    "Bebop piano solo, percussive accent, rapid chromatic runs, light bass backing",
    "Hard bop quintet, heavy rimshots, bluesy guitar solos, swinging bassline",
    "Post-bop piano trio waltz, delicate touch, complex polyrhythms, deep bass",
    "1940s bebop classic, tight arrangement, sudden tempo change, virtuosic solos",
    "Hard bop tenor sax feature, deep warm tone, soaring high notes, driving tempo",
    "Post-bop saxophone duo, un-accompanied counterpoint, complex harmonic motion",
    "Bebop drum feature, trading fours with horn section, crisp snare accents",
    "Hard bop trumpet solo, plunger mute expressiveness, slow blues groove",
    "Post-bop piano exploration, suspended chords, rubato introduction, dynamic swell",
    "1950s hard bop recording, vintage studio mic warmth, punchy drum sound",
    "Bebop guitar feature, fast single-note lines, subtle tube amplifier overdrive",
    "Hard bop organ feature, roaring Leslie speaker effect, deep bass pedals",
    "Post-bop modal trumpet, searching melody, open space, shimmer cymbals",
    "Bebop alto sax and trumpet head, ultra-fast unison, tight rhythm section kick",
    "Hard bop blues ballad, muted horn, weeping electric guitar, slow tempo",
    "Post-bop drum solo, polyrhythmic independence, tonal tom tuning, expressive",
    "1940s bebop piano trio, fast stride-influenced right hand, walking bass",
    "Hard bop horn section, punchy dynamic stabs, heavy swinging drums",
    "Post-bop double bass solo, extended techniques, bowed upper register, subtle piano",
    "Bebop vibraphone and piano duet, crystalline unison lines, fast tempo",
    "Hard bop tenor sax and trumpet harmony, rich warm interval voicing",
    "Post-bop quartet, metric shifting, sharp alto sax cuts, energetic drums",
    "1950s hard bop finale, roaring ensemble climax, driving hi-hat wash",
    "Bebop fast blues, rapid solo turns, energetic bass walking, crisp snare",
    "Hard bop piano solo, soulful blues rolls, percussive octave chords",
    "Post-bop soprano sax lead, soaring lyrical melody over complex rhythm",
    "Bebop trumpet feature, blazing high register, effortless speed",
    "Hard bop drum groove, heavy press roll, swinging ride, percussive bass",
    "Post-bop piano trio, angular melody, unexpected pause, driving bass burst",
    "1940s bebop live recording, energetic applause, burning tempo, fast horns",
    "Hard bop blues feature, gritty sax, swelling organ, slow driving beat",
    "Post-bop trumpet solo, dark woody tone, intricate intervallic jumps",
    "Bebop alto sax feature, soaring melody, rapid ornamentation, medium swing",
    "Hard bop quintet, driving minor blues theme, fiery solos, tight endings",
    "Post-bop ensemble, floating rhythm feel, lush non-functional harmony",

    # --- 241-320: Cool Jazz, West Coast & Modal Jazz ---
    "1950s Cool Jazz, muted Miles Davis style trumpet, relaxed rhythm, late night vibe",
    "West Coast cool jazz sextet, French horn, baritone sax, light brushed drum groove",
    "Modal jazz masterpiece, two-chord vamp, expansive soprano sax improvisation",
    "Cool jazz piano trio, delicate touch, sparse notes, smooth bass, soft cymbals",
    "1950s West Coast jazz, counterpoint horn arrangements, relaxed medium tempo",
    "Modal jazz exploration, static bass drone, searching tenor saxophone melody",
    "Cool jazz trumpet lead, soft cup mute tone, whisper-quiet brushed drums",
    "West Coast cool jazz flute feature, airy tone, gentle acoustic guitar backing",
    "Modal jazz waltz, 3/4 time signature, expansive piano chord voicings, rolling drums",
    "1950s cool jazz lounge, warm tube warmth, subtle reverb, relaxing atmosphere",
    "Cool jazz saxophone feature, smooth sub-tone, laid-back beat, quiet rimshots",
    "West Coast jazz chamber ensemble, cello, oboe, jazz guitar, soft bass",
    "Modal jazz piano solo, fourth-interval chord voicings, open spacious sound",
    "Cool jazz vibraphone lead, soft felt mallets, gentle tremolo, smooth tempo",
    "1950s cool jazz quartet, muted trumpet, clean archtop guitar, light bass",
    "Modal jazz bassline, hypnotic repeating motif, shimmering cymbal wash",
    "Cool jazz clarinet feature, smooth middle register, delicate piano chords",
    "West Coast cool jazz arrangement, intertwined trumpet and alto sax counterpoint",
    "Modal jazz trumpet solo, long sustained notes, deep emotional resonance",
    "Cool jazz guitar trio, soft thumb-picked chord melody, quiet upright bass",
    "1950s West Coast studio session, pristine mono sound, perfectly blended horns",
    "Modal jazz saxophone duet, harmony based on open fifths, meditative mood",
    "Cool jazz piano feature, block chords in locked-hands style, gentle swing",
    "West Coast cool jazz baritone sax lead, soft low-end rumble, brushed drums",
    "Modal jazz rhythm section, subtle metric shifts, wide open harmonized chords",
    "Cool jazz trumpet and flugelhorn duo, velvet smooth brass tone, slow pace",
    "1950s cool jazz ballad, whisper-soft brushed snare, long sustained horn notes",
    "Modal jazz piano exploration, pentatonic scale runs, deep acoustic resonance",
    "Cool jazz flute and muted trumpet lead, floating airy melody, light bass",
    "West Coast cool jazz arrangement, subtle polyphonic brass counterpoint",
    "Modal jazz tenor sax feature, contemplative mood, sparse piano accents",
    "Cool jazz double bass solo, soft wooden plucks, quiet cymbal tap accompaniment",
    "1950s cool jazz late-night session, relaxed groove, understated solos",
    "Modal jazz drums, emphasis on soft cymbal colors, light snare taps",
    "Cool jazz sax quartet, four-part smooth horn harmony, no rhythm section",
    "West Coast cool jazz trio, guitar, vibraphone, double bass, smooth interplay",
    "Modal jazz guitar solo, wide intervallic stretches, atmospheric delay effect",
    "Cool jazz flugelhorn solo, ultra-warm tone, romantic late-night melody",
    "1950s cool jazz broadcast, soft announcer intro, smooth trumpet lead",
    "Modal jazz ensemble, hypnotic bass groove, expansive horn improvisation",
    "Cool jazz piano solo, light stride elements slowed down, soft touch",
    "West Coast cool jazz brass quintet, French horn lead, gentle swinging beat",
    "Modal jazz soprano sax solo, haunting oriental-influenced scalar runs",
    "Cool jazz guitar chord melody, rich inner voice movement, warm room tone",
    "1950s cool jazz club, intimate ambient murmur, soft brushed snare drum",
    "Modal jazz piano chords, sustained open chords, lots of room ambiance",
    "Cool jazz alto sax feature, smooth lyrical lines, subtle vibraphone backing",
    "West Coast cool jazz arrangement, muted trombone and flute unison line",
    "Modal jazz trumpet exploration, minimal notes, maximum emotional expression",
    "Cool jazz rhythm section, steady soft walking bass, quiet hi-hat snaps",
    "1950s cool jazz radio recording, warm analog tape glow, smooth dynamics",
    "Modal jazz bass and drum duet, sparse bass motif, delicate brushwork",
    "Cool jazz vibraphone and guitar duet, soft crystalline tones, relaxed pace",
    "West Coast cool jazz ensemble, buoyant medium swing, polished horn lines",
    "Modal jazz tenor sax and trumpet harmony, parallel fourth voicings",
    "Cool jazz flugelhorn ballad, lush string backing, romantic slow tempo",
    "1950s cool jazz trio, piano, bass, brushed snare, perfectly balanced mix",
    "Modal jazz piano solo, sustained pedal point bass, soaring right hand",
    "Cool jazz saxophone duet, gentle harmonized lines, airy room tone",
    "West Coast cool jazz flute lead, light bossa-influenced rhythm, soft percussion",
    "Modal jazz trumpet lead, open horn, wide reverberant space, quiet drums",
    "Cool jazz double bass solo, melodic high-register playing, soft cymbal touches",
    "1950s cool jazz lounge track, smoky atmosphere, understated solos",
    "Modal jazz drum feature, atmospheric cymbal swells, soft mallet tom rolls",
    "Cool jazz muted trumpet, harmon mute sound, quiet piano comping",
    "West Coast cool jazz baritone sax and flute duet, low and high contrast",
    "Modal jazz piano trio, open pedal point, expansive dynamic build",
    "Cool jazz guitar lead, clean tone, light chorus effect, smooth swing",
    "1950s cool jazz quartet, velvety sax, soft piano, gentle bass plucks",
    "Modal jazz saxophone exploration, long modal scalar runs, steady drone",
    "Cool jazz flugelhorn feature, breathy sub-tone, slow romantic ballad",
    "West Coast cool jazz arrangement, counterpoint between muted brass and reed",
    "Modal jazz piano chords, slow harmonic shifting, deep spatial reverb",
    "Cool jazz double bass feature, soft arco bowing, gentle piano backing",
    "1950s cool jazz theme, memorable smooth horn melody, relaxed groove",
    "Modal jazz ensemble climax, intensity build over static bass drone",
    "Cool jazz sax quartet, gentle swing feel, rich lush dynamic blend",
    "West Coast cool jazz vibraphone solo, smooth mallet attack, soft bass",
    "Modal jazz trumpet and sax unison, wide expansive modal theme",
    "Cool jazz late-night ballad, muted trumpet, soft rain ambient sound background",

    # --- 321-400: Modern Fusion, Acid Jazz, City Pop & Nu-Jazz ---
    "1970s jazz fusion, fiery electric guitar solo, synth bass, complex time signature",
    "Modern jazz fusion, slap bass feature, tight pocket drums, Rhodes piano",
    "Acid jazz groove, funky Hammond organ, electric guitar wah-wah, hip-hop drum beat",
    "Japanese 80s city pop jazz fusion, slap bass, smooth synthesizers, guitar solo",
    "Nu-jazz track, broken beat electronic drums, live sax solo, sub-bass",
    "1970s jazz-rock fusion, distorted electric violin, heavy drum fills, clavinet",
    "Acid-funk jazz, talkbox vocals, bouncy synth bassline, tight brass stabs",
    "Modern fusion guitar solo, legato fast runs, overdrive pedal, spatial delay",
    "80s city pop jazz, smooth sax lead, bright digital synths, punchy bass drum",
    "Nu-jazz ensemble, electronic glitch beats, warm Fender Rhodes, live trumpet",
    "1970s jazz fusion keyboard solo, Minimoog synth lead, pitch bends, fast tempo",
    "Acid jazz guitar solo, funky single-note muted strumming, tight backbeat",
    "City pop jazz ballad, sweeping synth strings, smooth guitar solo, lush 80s reverb",
    "Nu-jazz saxophone feature, pitch-shifted sax effects, deep electronic sub-bass",
    "1970s fusion drum solo, double-kick accents, rapid tom runs, complex meter",
    "Acid jazz flute feature, funky vocal grunts, driving percussion, organ chords",
    "Modern fusion bass solo, thumb slap and pop, tap technique, chorus pedal",
    "Japanese city pop jazz intro, catchy brass fanfares, slap bass groove",
    "Nu-jazz chillout, warm analog synth pads, muted trumpet lead, electronic beat",
    "1970s jazz fusion band, electric piano, fretless bass, soprano sax, complex beat",
    "Acid jazz organ feature, Hammond B3 drive, Leslie speaker swirl, funk drums",
    "Modern fusion trumpet feature, wah pedal effect, delay trails, heavy drum beat",
    "80s city pop jazz fusion guitar duet, soaring distortion solos, clean rhythm",
    "Nu-jazz broken beat, syncopated electronic rimshots, live acoustic bass",
    "1970s fusion bass groove, envelope filter effect, syncopated funk drums",
    "Acid jazz brass section, tight percussive horn stabs, funky rhythm guitar",
    "Modern fusion piano solo, virtuosic jazz harmony over rock drum groove",
    "Japanese city pop jazz saxophone solo, ultra-smooth 80s production, chorus pedal",
    "Nu-jazz lounge, deep sub-bass, filtered piano samples, live trumpet solos",
    "1970s jazz-rock guitar battle, dual electric guitars, heavy organ rhythm",
    "Acid jazz Rhodes feature, dreamy chord extensions, relaxed hip-hop beat",
    "Modern fusion drum feature, metric modulation, crisp snare, heavy rimshots",
    "80s city pop jazz rhythm section, slap bass, electronic drum machine, funky synth",
    "Nu-jazz saxophone improvisation, electronic vocal chop background, heavy bass",
    "1970s fusion keyboard duet, acoustic piano and Fender Rhodes interplay",
    "Acid jazz bassline, heavy funk pocket, percussive muted guitar chops",
    "Modern fusion saxophone lead, aggressive tone, complex syncopated beat",
    "Japanese city pop jazz outro, fading synth solo, brass swells, slap bass line",
    "Nu-jazz trumpet feature, electronic delay trails, minimalist synth bass",
    "1970s jazz fusion track, odd meter 7/8 time, fast synthesizer arpeggios",
    "Acid jazz flute solo, rapid breathy runs, funky wah-wah guitar accompaniment",
    "Modern fusion electric guitar, screaming high notes, fast arpeggios, heavy drums",
    "80s city pop jazz piano solo, acoustic grand piano over synth bassline",
    "Nu-jazz electronic beat, crisp hi-hats, live sax wails, sub-bass pulse",
    "1970s jazz fusion fretless bass solo, vocal-like slides, chorus effect",
    "Acid jazz guitar feature, clean compressed tone, groovy organ comping",
    "Modern fusion quintet, electric guitar, sax, Rhodes, electric bass, drums",
    "Japanese city pop jazz funk, slap bass intro, bright brass section, uptempo",
    "Nu-jazz track, ambient synth textures, live brushed drum beat, muted horn",
    "1970s jazz-rock organ solo, distorted Hammond organ, fast rotary speaker",
    "Acid jazz beat, heavy vinyl kick drum, jazzy piano loop, live sax fills",
    "Modern fusion bass and guitar unison line, ultra-fast complex melody",
    "80s city pop jazz saxophone duet, alt and tenor harmony, lush 80s reverb",
    "Nu-jazz lounge beat, sidechained synth pads, live trumpet, deep bass",
    "1970s fusion saxophone feature, octave pedal effect, heavy funk groove",
    "Acid jazz percussive intro, shaker, congas, groovy bassline drop",
    "Modern fusion keys solo, Prophet synth patch, pitch wheel bends, fast tempo",
    "Japanese city pop jazz guitar rhythm, infectious disco-influenced funk strumming",
    "Nu-jazz acoustic bass and electronic beat blend, warm organic synths",
    "1970s jazz fusion acoustic guitar feature, fast nylon picking over funk drums",
    "Acid jazz horn stabs, syncopated brass punch, groovy electric bass",
    "Modern fusion drum and bass duo, complex polyrhythmic interplay",
    "80s city pop jazz ballad piano lead, romantic synth strings background",
    "Nu-jazz trumpet feature, harmon mute, electronic delay loops, sub-bass",
    "1970s fusion keyboard solo, percussive clavinet, envelope filter effect",
    "Acid jazz organ groove, deep bass pedals, snappy drum backbeat",
    "Modern fusion guitar lead, smooth overdrive, wide stereo ping-pong delay",
    "Japanese city pop jazz brass fanfare, vibrant 80s horn arrangements",
    "Nu-jazz broken beat drum solo, stutter effects, live acoustic cymbals",
    "1970s jazz fusion epic, dynamic shift from soft piano to heavy guitar fusion",
    "Acid jazz rhythm section, tight funk pocket, warm Rhodes chord comping",
    "Modern fusion bass solo, tapping techniques, crisp high-end punch",
    "80s city pop jazz groove, slap bass, digital synth brass, fast tempo",
    "Nu-jazz saxophone lead, processing effects, deep atmospheric synth drone",
    "1970s fusion drum groove, ghost notes, open hi-hat accents, heavy bass",
    "Acid jazz guitar solo, blues-inflected jazz phrasing over funk beat",
    "Modern fusion piano feature, fast modal runs over heavy funk drums",
    "Japanese city pop jazz synth solo, analog pitch bends, 80s production",
    "Nu-jazz track, lo-fi electronic filter, live sax solo, heavy sub-bass",
    "1970s jazz fusion climax, unison guitar and synth line, massive drum fills",

    # --- 401-480: Latin, Afro-Cuban, Bossa Nova & Samba Jazz ---
    "Upbeat Bossa Nova, nylon string guitar, light percussion, sunny Brazilian feel",
    "Afro-Cuban descarga, improvised mambo brass, driving congas, timbales solo",
    "Brazilian samba-jazz, fast acoustic nylon guitar, lively pandeiro, soaring flute",
    "Latin jazz quintet, bright piano montuno, heavy conga groove, energetic brass",
    "Bossa Nova ballad, gentle acoustic guitar chords, soft whispered vocal hums",
    "Afro-Cuban jazz saxophone feature, fiery alto sax over energetic timbales beat",
    "Samba-jazz piano solo, fast rhythmic basslines, sparkling right-hand runs",
    "Latin jazz trumpet lead, soaring high notes, driving 2-3 clave rhythm section",
    "Bossa Nova guitar feature, classic Stan Getz style sax melody, soft drums",
    "Afro-Cuban piano montuno feature, percussive octave chords, heavy bass tumbao",
    "Samba-jazz drum feature, surdo bass drum, crisp tamborim, fast rimshots",
    "Latin jazz flute solo, rapid staccato runs, driving guiro and congas",
    "Bossa Nova piano trio, smooth nylon guitar backing, soft brushed drum beat",
    "Afro-Cuban brass section, dynamic trumpet and trombone stabs, fast tempo",
    "Samba-jazz saxophone duet, soaring tenor lines, frantic samba percussion",
    "Latin jazz trombone feature, rich warm slides over heavy Latin percussion",
    "Bossa Nova waltz in 3/4 time, nylon guitar, soft flute melody, gentle rhythm",
    "Afro-Cuban timbales drum solo, cascading rim clicks, ringing cowbells, fast tempo",
    "Samba-jazz acoustic guitar solo, virtuosic thumb-and-finger picking, bright tone",
    "Latin jazz piano trio, intricate montuno patterns, swinging bass tumbao",
    "Bossa Nova sax feature, smooth tenor sub-tone, airy lounge atmosphere",
    "Afro-Cuban conga solo, slap and open tones, complex syncopated polyrhythms",
    "Samba-jazz flute lead, breezy tropical melody, fast pandeiro beat",
    "Latin jazz vibraphone solo, bright mallet accents over fast salsa beat",
    "Bossa Nova guitar chord melody, classic Jobim voicings, smooth vocal breath",
    "Afro-Cuban jazz ensemble, heavy mambo brass, thunderous percussion break",
    "Samba-jazz big band, massive horn section, energetic Brazilian percussion",
    "Latin jazz trumpet duo, harmonized brass lines, driving bongos and congas",
    "Bossa Nova duet, soft acoustic guitar and gentle vibraphone, sunny mood",
    "Afro-Cuban piano feature, percussive chord blocks, fast clave timing",
    "Samba-jazz piano trio, rapid 6/8 to 2/4 rhythm shift, energetic bass",
    "Latin jazz guitar solo, clean archtop, fast octave runs over Latin beat",
    "Bossa Nova lounge, subtle ocean wave ambient sounds, gentle guitar strum",
    "Afro-Cuban trombone section, heavy low-end brass riffs, tight timbales",
    "Samba-jazz saxophone feature, bright soaring soprano sax, fast tempo",
    "Latin jazz percussion ensemble, congas, bongos, timbales, claves in unison",
    "Bossa Nova flute solo, gentle breathy tone, relaxed acoustic guitar",
    "Afro-Cuban trumpet solo, soaring high register, fast salsa rhythm",
    "Samba-jazz double bass solo, syncopated thumb plucks, lively percussion",
    "Latin jazz piano montuno intro, swelling brass stabs, fast Latin groove",
    "Bossa Nova guitar solo, delicate fingerstyle, warm nylon string sound",
    "Afro-Cuban jazz waltz, 3/4 clave rhythm, expressive saxophone lead",
    "Samba-jazz vibraphone feature, shimmering mallet rolls, fast Brazilian beat",
    "Latin jazz alto sax feature, fiery bebop lines over heavy conga beat",
    "Bossa Nova vocal scat duet, soft melodic unison with acoustic guitar",
    "Afro-Cuban bongo solo, fast finger rolls, pitch-bent drumhead sounds",
    "Samba-jazz rhythm section, heavy surdo pulse, fast shaking chocalho",
    "Latin jazz brass fanfares, aggressive salsa horn section, fast tempo",
    "Bossa Nova trio, piano lead, gentle nylon guitar rhythm, soft brushed snare",
    "Afro-Cuban piano and bass duet, tight montuno and tumbao alignment",
    "Samba-jazz acoustic guitar and flute duet, complex Brazilian counterpoint",
    "Latin jazz tenor sax solo, smoky tone, fast Latin blues phrasing",
    "Bossa Nova electric guitar feature, clean hollow-body, soft bossa beat",
    "Afro-Cuban jazz big band climax, roaring brass stabs, wild timbales roll",
    "Samba-jazz piano solo, syncopated left-hand bass, fast right-hand trills",
    "Latin jazz vibraphone and flute duet, crystalline unison lines, conga beat",
    "Bossa Nova sunset vibe, soft acoustic guitar, gentle bass, warm room reverb",
    "Afro-Cuban brass battle, two trumpets trading fast high-note fills",
    "Samba-jazz percussion break, fast cuica wails, pandeiro snaps, energetic",
    "Latin jazz piano feature, intricate montuno transitions, heavy clave beat",
    "Bossa Nova solo acoustic guitar, classic syncopated thumb pattern, warm room",
    "Afro-Cuban saxophone quartet, four-part Latin horn harmonies, no drums",
    "Samba-jazz flute solo, rapid flutter-tonguing, driving samba rhythm",
    "Latin jazz double bass feature, bow and pluck combination, conga backing",
    "Bossa Nova piano lead, soft chordal accents, subtle brushed cymbals",
    "Afro-Cuban timbales and conga duet, intricate polyrhythmic conversation",
    "Samba-jazz big band horn section, punchy dynamic brass, fast tempo",
    "Latin jazz guitar feature, fiery nylon string solos, heavy clave beat",
    "Bossa Nova sax lead, soft soprano saxophone, warm sunny acoustic room",
    "Afro-Cuban piano montuno exploration, dynamic volume swells, heavy bass",
    "Samba-jazz acoustic bass solo, fast woody plucks, pandeiro accompaniment",
    "Latin jazz trumpet solo, plunger mute wah effects over salsa beat",
    "Bossa Nova trio, soft Fender Rhodes, nylon guitar, gentle brushes",
    "Afro-Cuban jazz quintet, fiery sax, montuno piano, driving percussion",
    "Samba-jazz vibraphone lead, fast mallet arpeggios, energetic drumbeat",
    "Latin jazz flute feature, intricate classical-influenced runs, conga beat",
    "Bossa Nova ballad guitar solo, soft expressive bends, warm nylon tone",
    "Afro-Cuban brass section, dynamic low trombone stabs, fast clave beat",
    "Samba-jazz piano and bass duet, fast syncopated Brazilian groove",
    "Latin jazz finale, full band jam, screaming trumpet, wild timbales solo",

    # --- 481-560: Soul Jazz, Organ Trios, Funk & Blues Jazz ---
    "Soul-jazz, groovy Hammond organ, bluesy electric guitar, punchy drum backbeat",
    "Classic organ trio, Hammond B3 drawbar organ, hollow-body electric guitar, brushes",
    "Kansas City jump blues jazz, honking tenor sax, boogie-woogie piano, swing",
    "Gospel-infused soul jazz, warm Fender Rhodes, swelling B3 organ, bluesy phrasing",
    "Groovy soul-jazz tenor sax lead, gritty tone, blues bends, heavy backbeat",
    "Organ trio blues, 12-bar form, screaming Hammond B3, rhythmic guitar comps",
    "Slow tempo blues-jazz ballad, expressive electric guitar bends, late-night bar vibe",
    "Soul-jazz drum groove, heavy snare backbeat, swinging hi-hat, deep kick drum",
    "Gospel-jazz piano solo, churchy chord voicings, tremolo rolls, soulful feel",
    "Funk-jazz guitar feature, envelope filter, syncopated rhythm, heavy bass groove",
    "Soul-jazz alto sax solo, bluesy scoops, percussive articulation, driving beat",
    "Hammond B3 organ solo, fast rotary speaker acceleration, percussive click attack",
    "Jump blues sax section, honking baritone sax, energetic riffing, fast swing",
    "Soul-jazz trumpet feature, open horn blues phrasing, groovy bassline",
    "Gospel-jazz organ feature, swelling B3 pedal notes, bluesy piano chords",
    "Electric jazz blues guitar solo, warm tube amp overdrive, expressive bends",
    "Soul-jazz quintet, catchy soulful horn riff, deep pocket drum beat",
    "Organ trio funk, heavy bass pedals, fast guitar thumb-picking, tight drums",
    "Slow soul-jazz ballad, expressive tenor sax sub-tone, gentle organ swells",
    "Gospel-jazz vocal and piano duet, expressive bluesy chords, warm atmosphere",
    "Soul-jazz guitar feature, thumb-picked octave solos, swinging organ backing",
    "Hammond B3 drawbar solo, screaming high notes, percussive left-hand bass",
    "Jump blues piano solo, rolling boogie-woogie left hand, fast right hand blues",
    "Soul-jazz flute feature, gritty vocalized flute technique, groovy drum beat",
    "Gospel-jazz ensemble, choir-like brass harmonies, uplifting soul-jazz beat",
    "Electric blues-jazz guitar duet, trading bluesy fills, slow barroom tempo",
    "Soul-jazz saxophone and organ unison, catchy blues motif, driving drums",
    "Organ trio jam session, fast tempo blues, burning organ solos, energetic",
    "Soul-jazz drum solo, heavy rimshots, funky snare rolls, driving kick drum",
    "Gospel-jazz Fender Rhodes feature, warm chorus effect, bluesy chord rolls",
    "Funk-jazz bass solo, heavy slap and pop, tight pocket drum backing",
    "Soul-jazz trumpet solo, plunger mute growls, greasy organ chords",
    "Hammond B3 Leslie speaker feature, fast rotation acceleration, warm tube drive",
    "Jump blues tenor sax feature, aggressive screaming high notes, fast tempo",
    "Soul-jazz guitar solo, clean archtop tone, bluesy slide and bend technique",
    "Gospel-infused jazz piano trio, churchy cadence chords, swinging bassline",
    "Slow 12-bar blues jazz ballad, muted trumpet, weeping electric guitar, slow beat",
    "Soul-jazz organ feature, percussive bassline, fast right-hand blues runs",
    "Funk-jazz brass section, tight punchy horn accents, heavy funk backbeat",
    "Soul-jazz saxophone feature, deep smoky tenor tone, bluesy inflections",
    "Gospel-jazz organ and guitar interplay, rich soulful call-and-response",
    "Organ trio slow blues, deep bass pedals, subtle guitar comping, brushed drums",
    "Soul-jazz flute solo, overblown percussive technique, funky bassline",
    "Electric blues-jazz guitar feature, slow expressive bends, warm amp tremolo",
    "Soul-jazz trumpet and tenor sax harmony, rich warm blues voicings",
    "Hammond B3 drawbar exploration, soft low-end rumble shifting to screaming highs",
    "Jump blues big band, roaring horn section, fast boogie-woogie beat",
    "Soul-jazz rhythm section, rock-solid funk-jazz pocket, steady hi-hat",
    "Gospel-jazz piano feature, rich extended chords, emotional slow tempo",
    "Funk-jazz guitar strumming, syncopated muted chops, heavy bass groove",
    "Soul-jazz baritone sax lead, heavy low-end blues riffs, punchy drums",
    "Organ trio groove, bouncing bass pedals, rhythmic guitar chords, crisp snare",
    "Slow soul-jazz ballad, muted trombone lead, gentle organ backing, soft beat",
    "Gospel-jazz saxophone solo, soaring soulful notes, uplifting chord progression",
    "Electric blues-jazz guitar lead, clean tone, bluesy chord melody, slow tempo",
    "Soul-jazz quintet, catchy repeating horn hook, heavy swinging drum beat",
    "Hammond B3 bass pedal feature, deep room-shaking low end, tight drum pocket",
    "Jump blues guitar solo, sharp biting tone, fast swing rhythm backing",
    "Soul-jazz alto sax lead, bluesy scoops, bright ringing tone, funky beat",
    "Gospel-jazz piano and organ duet, rich harmonized chord rolls, uplifting mood",
    "Funk-jazz bassline, heavy synth-bass overlay, tight syncopated drum groove",
    "Soul-jazz guitar feature, Wes Montgomery style thumb picking, warm tone",
    "Organ trio fast tempo blues, rapid organ runs, energetic guitar solos",
    "Slow soul-jazz trumpet ballad, soft cup mute, gentle organ swells, slow drums",
    "Gospel-jazz brass section, warm three-part horn harmony, steady soul beat",
    "Electric blues-jazz guitar solo, expressive vibrato, tube amp saturation",
    "Soul-jazz flute lead, breezy bluesy runs, groovy organ backing",
    "Hammond B3 organ solo, rapid chord stabs, percussive click attack",
    "Jump blues sax duet, tenor and baritone sax counterpoint, fast beat",
    "Soul-jazz drum feature, percussive tom fills, funky snare backbeat",
    "Gospel-jazz Fender Rhodes solo, warm bell-like tine tones, bluesy runs",
    "Funk-jazz horn section, ultra-tight dynamic stabs, heavy funk groove",
    "Soul-jazz tenor sax solo, smoky sub-tone, soulful blues bends, slow tempo",
    "Organ trio ballad, soft drawbar organ tones, gentle guitar, brushed drums",
    "Slow soul-jazz guitar lead, deep expressive bends, warm room reverb",
    "Gospel-jazz ensemble climax, soaring saxophone over swelling organ and drums",
    "Soul-jazz trumpet solo, open horn power, soulful blues inflections",
    "Hammond B3 organ groove, fast Leslie speaker swirl, punchy drums",
    "Funk-jazz guitar rhythm, wah-wah pedal effect, syncopated funk beat",
    "Soul-jazz finale, full ensemble blues jam, screaming organ, fast drums",

    # --- 561-640: Spiritual, Free, Avant-Garde & World Jazz ---
    "Spiritual jazz, meditative saxophone drones, shimmering chimes, cosmic atmosphere",
    "Avant-garde free jazz, dissonant piano, chaotic rhythms, experimental solos",
    "Ethio-jazz, hypnotic pentatonic scales, vintage Farfisa organ, brass stabs",
    "Japanese spiritual jazz, bamboo shakuhachi flute, harp, freeform percussion",
    "Dark ambient post-jazz, bowed cymbals, pitch-shifted sax wails, slow drone",
    "Cape Town jazz, bright marimba rhythms, uplifting horn arrangements, coastal feel",
    "Spiritual jazz harp solo, cascading arpeggios, shimmering gongs, ambient space",
    "Free jazz sax solo, overblowing, multiphonics, un-metered drum chaos",
    "Ethio-jazz sax feature, pentatonic modal melody, hypnotic bassline, brass fills",
    "Indo-jazz fusion, sitar improvisation, tabla polyrhythms, jazz saxophone",
    "Spiritual jazz tenor sax lead, soaring emotional cries, rolling cymbal swells",
    "Avant-garde piano solo, cluster chords, prepared piano strings, microtonal",
    "Ethio-jazz organ groove, Farfisa organ wails, dark pentatonic modal feel",
    "Flamenco jazz guitar, fast rasgueado strumming, cajon rhythm, jazz sax lead",
    "Spiritual jazz drum feature, atmospheric mallet tom rolls, shimmering bells",
    "Free jazz double bass solo, harsh arco bowing, snapping string plucks",
    "Ethio-jazz brass section, sharp staccato stabs, hypnotic bass guitar",
    "Middle Eastern jazz fusion, oud solo, frame drum rhythm, jazz trumpet",
    "Spiritual jazz ensemble, meditative bass drone, soaring soprano sax",
    "Avant-garde free jazz trumpet solo, extended techniques, flutter tonguing",
    "Ethio-jazz flute solo, breathy pentatonic scales, vintage organ backing",
    "Celtic jazz fusion, wooden Irish flute, jazz acoustic bass, bodhran beat",
    "Spiritual jazz piano solo, repetitive hypnotic chords, swelling volume",
    "Free jazz sax duet, two saxophones shouting and overlapping in dense chaos",
    "Ethio-jazz drum groove, syncopated 6/8 rhythm, heavy vintage bass",
    "Gipsy-oriental jazz fusion, fast accordion solos, darbuka drum, jazz sax",
    "Spiritual jazz flute feature, meditative breath sounds, cosmic synth pad",
    "Avant-garde free jazz quartet, total improvisation, sharp dynamic shifts",
    "Ethio-jazz tenor sax feature, deep smoky pentatonic runs, organ chords",
    "Afro-beat jazz fusion, polyrhythmic brass section, heavy bassline, congas",
    "Spiritual jazz double bass drone, bow harmonic overtones, bell tree chimes",
    "Free jazz drum solo, completely unmetered, percussive cymbal explosions",
    "Ethio-jazz trumpet feature, haunting modal melodies, sharp brass stabs",
    "Moroccan Gnawa jazz fusion, guembri bassline, qraqeb iron castanets, sax",
    "Spiritual jazz ensemble climax, intense emotional wall of sound, soaring sax",
    "Avant-garde piano and sax duet, angular dissonant dialogue, free tempo",
    "Ethio-jazz guitar feature, muted rhythmic picking, hypnotic organ loop",
    "Balkan jazz fusion, 11/8 complex time signature, fast clarinet solos",
    "Spiritual jazz soprano sax exploration, circular breathing, sustained notes",
    "Free jazz collective improvisation, all instruments playing simultaneously",
    "Ethio-jazz vibraphone lead, dark pentatonic arpeggios, hypnotic bass",
    "Nordic spiritual jazz, atmospheric trumpet, glacial delays, acoustic bass",
    "Spiritual jazz harp and flute duet, ethereal cascading sounds, gong hits",
    "Avant-garde drum feature, percussive metal object sounds, bowed cymbals",
    "Ethio-jazz big band, massive pentatonic horn section, driving vintage drums",
    "Indian tabla and jazz piano duo, complex rhythmic cycles, modal chords",
    "Spiritual jazz tenor sax ballad, deep emotional warmth, rolling cymbals",
    "Free jazz trombone solo, aggressive plunger mute growls, erratic bass",
    "Ethio-jazz alto sax solo, rapid pentatonic runs, sharp Farfisa chords",
    "Andean jazz fusion, charango guitar picking, pan flute, jazz upright bass",
    "Spiritual jazz piano chords, rich open voicings, long sustain pedal",
    "Avant-garde bass solo, extended bow techniques, percussive body taps",
    "Ethio-jazz organ feature, vintage organ vibrato, dark hypnotic rhythm",
    "West African jazz, kora harp melody, warm acoustic bass, sax harmonies",
    "Spiritual jazz ensemble, peaceful meditative opening, slow dynamic build",
    "Free jazz trumpet and drums duo, high speed chaotic dialogue",
    "Ethio-jazz saxophone and trumpet unison, iconic pentatonic melody",
    "Turkish jazz fusion, ney flute solo, darbuka percussion, jazz piano chords",
    "Spiritual jazz percussion feature, rainsticks, gongs, shakers, warm bass",
    "Avant-garde vocal jazz, freeform vocal improvisation, unconventional sounds",
    "Ethio-jazz bassline, repeating hypnotic groove, crisp vintage drums",
    "Flamenco-jazz sax feature, passionate Andalusian scales, cajon backbeat",
    "Spiritual jazz saxophone drone, low sustained notes, shimmering percussion",
    "Free jazz piano explosion, rapid full-keyboard clusters, wild drum chaos",
    "Ethio-jazz guitar solo, clean electric guitar, bluesy pentatonic bends",
    "Japanese shakuhachi and jazz guitar duet, meditative atmospheric space",
    "Spiritual jazz piano trio, open modal chords, rolling cymbals, bass drone",
    "Avant-garde free jazz outro, fading chaotic textures, solitary sax cry",
    "Ethio-jazz flute and organ duet, haunting modal counterpoint",
    "Brazilian spiritual jazz, berimbau bow drone, flute lead, subtle bass",
    "Spiritual jazz trumpet lead, long sustained notes, cosmic synth drone",
    "Free jazz double bass and saxophone duo, rapid angular interplay",
    "Ethio-jazz brass stabs, syncopated horn punch, heavy vintage bassline",
    "World jazz percussion ensemble, multi-cultural rhythms, jazz flute solo",
    "Spiritual jazz finale, soothing meditative harp and sax resolution",
    "Avant-garde ensemble, microtonal brass harmonies, unpredictable rhythms",
    "Ethio-jazz tenor sax solo, haunting pentatonic phrasing, dark organ",
    "Indo-jazz flute solo, Indian classical ragas over jazz chord drone",
    "Spiritual jazz soprano sax lead, soaring spiritual intensity, gongs",
    "Ethio-jazz ensemble climax, driving pentatonic groove, wild brass solos",

    # --- 641-720: Solo Instruments, Trios & Unique Chamber Groupings ---
    "Solo jazz piano, complex chords, stride style, no other instruments",
    "Solo double bass performance, plucking and bowing techniques, deep resonance",
    "Solo gypsy jazz guitar, fast picking, acoustic wood body sound",
    "Solo jazz harp, ethereal extended chord voicings, cascading arpeggios",
    "Solo acoustic guitar, fingerstyle jazz chord melody, rich natural resonance",
    "Solo jazz trumpet, expressive open horn melody, natural room acoustics",
    "Solo tenor saxophone, smoky tone, un-accompanied ballad, soft room echo",
    "Solo vibraphone, lush four-mallet chord voicings, warm tremolo motor",
    "Solo jazz organ, Hammond B3 drawbars, bass pedals and right hand melody",
    "Solo jazz flute, breathy tone, percussive key clicks, natural hall acoustics",
    "Solo electric jazz guitar, clean hollow-body tone, rich chord voice-leading",
    "Solo baritone saxophone, deep low-end resonance, solitary melody",
    "Solo jazz drum kit performance, expressive dynamics, cymbal tonal colors",
    "Solo accordion jazz, French gypsy swing style, acoustic bellows sound",
    "Solo alto saxophone, bright soaring tone, virtuosic unaccompanied runs",
    "Acoustic jazz trio, double bass, piano, and brushed drums",
    "Organ trio, Hammond B3, hollow-body electric guitar, drum kit",
    "Gypsy swing jazz ensemble, dual acoustic guitars, upright bass, violin solo",
    "Vibraphone-led jazz quartet, crystalline mallet tones, swinging bass",
    "Guitar trio, electric archtop guitar, double bass, subtle brushed snare",
    "Piano trio ballad, delicate grand piano, soft bass plucks, brushed cymbals",
    "Saxophone trio, tenor sax, double bass, drums, wide open spacious feel",
    "Trumpet trio, muted trumpet, acoustic bass, brushed snare drum",
    "Flute trio, jazz flute, double bass, light percussion, breezy tempo",
    "Clarinette trio, warm woodwind lead, upright bass, gentle brush rhythm",
    "Bass clarinet-led chamber ensemble, deep woody notes, subtle cello backing",
    "Chamber jazz quintet, classical string quartet accompanied by jazz flute",
    "Jazz harp and cello duet, intertwining lines, gentle room acoustics",
    "Oboe and jazz piano duo, elegant woodwind melody, rich chord voicings",
    "French horn lead cool jazz sextet, soft brass harmonies, light rhythm",
    "String trio and jazz saxophone, sweeping violins, expressive sax solo",
    "Woodwind jazz ensemble, flute, oboe, clarinet, bassoon, jazz bass",
    "Marimba and vibraphone duo, sparkling mallet interplay, warm acoustic",
    "Accordion and double bass duo, Parisian swing, intimate acoustic space",
    "Violin and archtop guitar jazz duo, fast swing counterpoint, warm tone",
    "Brass trio, trumpet, French horn, trombone playing soft jazz counterpoint",
    "Solo piano rubato introduction, slow transition into swinging jazz trio",
    "Solo jazz bass, extended arco bowing melody, deep resonant wood tone",
    "Solo hollow-body guitar, fingerstyle jazz blues, warm tube amplifier",
    "Solo soprano saxophone, lyrical melody, pure crystalline tone",
    "Solo jazz trombone, velvet smooth tone, expressive slide vibrato",
    "Piano trio fast tempo, rapid piano arpeggios, burning walking bassline",
    "Organ trio funk groove, heavy Leslie organ, choppy electric guitar",
    "Gypsy jazz trio, rhythm acoustic guitar, upright bass, fast solo violin",
    "Vibraphone trio, sparkling mallets, acoustic bass, subtle snare drum",
    "Guitar trio waltz, 3/4 time signature, clean guitar tone, brushed drums",
    "Piano trio modal exploration, static bass drone, sweeping piano chords",
    "Saxophone trio fast bop, burning tenor sax, lightning fast acoustic bass",
    "Trumpet trio cool jazz, soft muted horn, gentle bass, quiet rimshots",
    "Flute trio bossa nova, breezy flute melody, nylon guitar, light percussion",
    "Solo piano stride stomp, energetic left hand, fast right-hand blues runs",
    "Solo electric guitar chord melody, modern jazz voicings, subtle reverb",
    "Solo double bass, rapid pizzicato walking bassline, resonant low end",
    "Solo tenor sax ballad, smoky late-night tone, expressive pitch bends",
    "Solo vibraphone ballad, sustained four-mallet chords, warm tremolo",
    "Chamber jazz trio, flute, cello, acoustic guitar, delicate interplay",
    "String quartet with jazz piano, rich classical textures, jazz swing",
    "Duet, soprano sax and acoustic guitar, airy melody, light strumming",
    "Duet, bass clarinet and double bass, deep woody low-end conversation",
    "Duet, vibraphone and piano, sparkling twin keyboard arpeggios",
    "Duet, muted trumpet and piano, soft late-night intimate ballad",
    "Duet, tenor sax and drums, raw energetic interplay, fast tempo",
    "Duet, electric guitar and Hammond organ, soulful bluesy call-and-response",
    "Duet, vocal scat and double bass, rhythmic vocal improvisations",
    "Solo jazz clarinet, liquid tone, unaccompanied blues improvisation",
    "Solo jazz harp ballad, lush sweeping chords, romantic mood",
    "Solo baritone sax blues, gritty low-end riffing, solo performance",
    "Solo acoustic bass waltz, melodic high-register playing in 3/4 time",
    "Solo jazz organ ballad, warm drawbar swells, subtle pedal bass",
    "Piano trio soul-jazz, churchy piano chords, steady swinging beat",
    "Guitar trio lounge, smooth electric guitar, soft bass, quiet brushed drums",
    "Gypsy jazz guitar duo, virtuosic soloing over fast rhythmic strumming",
    "Vibraphone trio ballad, shimmering mallets, soft bass, brushed cymbals",
    "Saxophone trio modal, searching tenor sax, interactive bass and drums",
    "Chamber jazz ensemble, woodwinds, strings, jazz rhythm section, elegant",
    "Duet, flute and double bass, playful counterpoint, light acoustic space",
    "Duet, violin and piano, fast swing lines, energetic performance",
    "Duet, trombone and piano, warm velvet brass tone, rich piano chords",
    "Duet, alto sax and piano, expressive ballad, soft romantic mood",
    "Solo jazz piano coda, dramatic final chords, long natural decay",

    # --- 721-800: Atmospheres, Historical Textures & Environments ---
    "Dark, noir jazz, solo trumpet with heavy reverb, rainy street atmosphere",
    "Intimate jazz club recording, sounds of clinking glasses and distant chatter",
    "Street performer playing a saxophone in a subway tunnel, metallic echoes",
    "Low-fidelity bedroom jazz, muffled sound, warm tape hiss",
    "Jazz band playing in a massive stone cathedral, long echo tails",
    "Vintage cassette tape demo, saturated audio compression, warm analog hiss",
    "Sunlit outdoor park performance, ambient birds chirping, gentle breeze, acoustic trio",
    "High-end audiophile studio session, pristine instrument isolation, dynamic punch",
    "Empty auditorium sound check, booming acoustic reflections, solitary trumpet calls",
    "Underground speakeasy, murmur of crowd, ice clinking, unamplified acoustic jazz",
    "1970s radio studio broadcast, tube preamp warmth, dynamic FM radio compression",
    "Late-night radio ballad, lush muted horn section, gentle brushed snare, romantic atmosphere",
    "1920s scratchy vinyl record, shellac 78rpm needle noise, thin bandpass filter",
    "Late-night rain on window, distant siren, smoky noir saxophone, heavy reverb",
    "Smoky 1950s basement jazz club, muted trumpet, soft applause, clinking glasses",
    "Outdoor summer jazz festival, crowd cheering background, warm ambient air",
    "Subway station saxophonist, booming concrete acoustics, distant train rumble",
    "Vintage 1940s mono radio broadcast, mild static, warm tube compression",
    "Pristine audiophile jazz recording, ultra-wide soundstage, silent noise floor",
    "Lo-fi vinyl jazz, heavy crackle and pop, muffled drums, sampled piano loop",
    "Old wooden church recording, natural acoustic reverb, solitary trumpet wail",
    "Open-air cafe jazz trio, distant traffic murmur, clinking coffee cups, sunshine feel",
    "Late-night bedroom recording, cassette deck flutter, soft electric guitar chord",
    "Cozy fireplace jazz lounge, crackling fire sounds, soft piano ballad",
    "Midnight street corner saxophonist, distant city traffic, long reverb tail",
    "1930s dance hall recording, muffled room sound, booming kick drum, tape hiss",
    "Audiophile piano trio session, razor-sharp instrument imaging, deep bass punch",
    "Speakeasy hidden bar, hushed crowd chatter, ice shaking in cocktail shaker, jazz quartet",
    "Analog tape warm recording, harmonic saturation, gentle wow and flutter, smooth sax",
    "Rainy evening jazz lounge, soft thunder in distance, muted horn, slow brushed drums",
    "Cathedral acoustics, solitary soprano saxophone, immense five-second reverb tail",
    "Subway tile tunnel acoustics, raw unamplified brass, echoing footstep sounds",
    "1960s Japanese jazz club recording, tape warmth, intimate room acoustics, clinking glasses",
    "Lo-fi boom-bap jazz beat, dusty vinyl texture, vinyl hiss, chill muted trumpet",
    "Sunlit patio acoustic jazz trio, gentle wind chimes, outdoor natural ambiance",
    "Vintage AM radio broadcast, low fidelity, bandpass filter, faint static hiss",
    "Empty recording studio live room, subtle room mic bleed, crisp drum acoustics",
    "Late-night jazz radio station, soft voiceover intro fading into smooth muted trumpet",
    "Smoky velvet-draped lounge, deep bass plucks, soft brushed snare, warm mood",
    "Outdoor park pavilion jazz band, distant laughter, ambient breeze, acoustic instruments",
    "1920s phonograph recording texture, narrow frequency response, surface crackle",
    "Rainy night street lamp ambiance, weeping electric jazz guitar, long hall delay",
    "Underground cave recording, unique acoustic stone reflection, solitary flute",
    "Audiophile live album, perfectly captured crowd applause, crystal clear instrument separation",
    "Lo-fi bedroom piano, soft pedal felt noises, cassette tape saturation, relaxed beat",
    "1950s radio studio session, vintage Neumann condenser mic warmth, smooth trumpet",
    "Cozy winter cabin jazz, crackling wood stove, slow acoustic guitar duet",
    "Subway platform jazz trio, echoing concrete walls, distant train arrivals",
    "Late-night cocktail lounge, low murmur of conversation, soft piano comping",
    "Vintage cassette deck tape bounce, dynamic saturation, warm analog hiss, jazz sax",
    "Pristine studio chamber acoustics, natural wooden room reflections, string quartet with sax",
    "Rooftop sunset jazz session, soft city wind, distant traffic hum, smooth bossa nova",
    "1930s shellac disc recording, strong needle hiss, fast syncopated stride piano",
    "Smoky film noir alleyway, solitary muted trumpet, distant rain puddles, heavy reverb",
    "Audiophile drum session, thunderous kick drum dynamics, shimmering pristine cymbals",
    "Lo-fi jazz-hop track, vinyl dust crackle, warm sampled Rhodes chord loops",
    "Empty theater rehearsal, expansive stage acoustics, solitary sax melody",
    "1970s analog tape master, rich saturation, punchy drum backbeat, groovy organ",
    "Outdoor beach bar jazz, distant ocean waves crashing, gentle bossa guitar",
    "Underground jazz cellar, heavy low-end bass resonance, dark intimate vibe",
    "1940s radio transcription disc, subtle turntable rumble, warm brass section",
    "Late-night rainstorm outside window, soft piano ballad indoors, peaceful atmosphere",
    "Audiophile double bass recording, clear finger-pluck attack, rich body resonance",
    "Lo-fi cassette jazz loop, heavy wow and flutter, warm muffled drum beats",
    "Subway corridor trumpet performance, metallic echo, distant reverberating footsteps",
    "1950s Hollywood soundstage recording, sweeping lush strings, crisp muted horn lead",
    "Cozy bookshelf studio, tight room treatment, dry intimate sax tone",
    "Rooftop jazz party, background crowd ambiance, bright horns, uplifting groove",
    "1920s gramophone recording, vintage shellac noise, horn band swing",
    "Rainy film noir detective office, dark solo sax melody, distant rain and thunder",
    "Audiophile vibraphone session, crystalline tone decay, wide stereo pan",
    "Lo-fi chillhop jazz, sampled piano, dusty vinyl crackle, slow relaxed tempo",
    "Large stone church, soaring soprano sax wails, six-second reverberation decay",
    "1960s analog master tape, warm saturated sound, punchy acoustic bass, clear cymbals",
    "Sunlit garden jazz quartet, chirping birds ambient background, gentle breeze, smooth sax",
    "Vintage shortwave radio jazz, drifting signal frequency, subtle static noise",
    "Intimate basement speakeasy, ice clinking in glass, unamplified acoustic piano trio",
    "Late-night urban balcony, distant city highway hum, solitary trumpet improvisation",
    "Audiophile big band recording, explosive dynamic range, razor-sharp horn stabs",
    "Lo-fi jazz outro, fading vinyl crackle, tape hiss, solitary piano chord decay"
]
chosen_numbers = list(range(1, len(prompts)+1))
"""
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
"""

import csv

# ==============================================================================
# 4. Main Execution Pipeline Across All Directories
# ==============================================================================
if __name__ == "__main__":
    TXT_OUTPUT = "evaluation_results.txt"
    CSV_OUTPUT = "evaluation_results.csv"
    CSV_FIELDS = ["folder", "count", "clap_score", "clap_fad", "panns_div_mean", "panns_div_std", "cond_vendi"]

    # Initialize output files so results are saved incrementally after each folder
    with open(TXT_OUTPUT, "w", encoding="utf-8") as f:
        f.write("EVALUATION RESULTS LOG\n")
        f.write("=" * 70 + "\n")

    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

    # --- Step A: Load Models Once ---
    print(f"\nLoading CLAP model on device: {device}...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False)
    clap_model.load_ckpt(model_id=1)
    clap_model.to(device)

    print("Loading PANNs AudioTagging model...")
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
        print(f"[{dir_idx}/{len(dir_list)}] Processing: {folder_name}...")

        if not os.path.exists(gen_dir):
            with open(TXT_OUTPUT, "a", encoding="utf-8") as f:
                f.write(f"\n[Error] Directory does not exist: {gen_dir}\n")
            continue

        raw_audio_paths = []
        raw_prompts = []
        for k in chosen_numbers:
            file_path = os.path.join(gen_dir, f"audioldm2_{k}.wav")
            if os.path.exists(file_path):
                raw_audio_paths.append(file_path)
                raw_prompts.append(prompts[k - 1])

        all_audio_paths = []
        chosen_prompts = []
        for path, prompt in zip(raw_audio_paths, raw_prompts):
            if len(filter_valid_audio_files([path])) > 0:
                all_audio_paths.append(path)
                chosen_prompts.append(prompt)

        if len(all_audio_paths) == 0:
            with open(TXT_OUTPUT, "a", encoding="utf-8") as f:
                f.write(f"\n[Error] No valid audio files found in directory: {gen_dir}\n")
            continue

        # 1. Compute CLAP Score (Text-Audio Alignment)
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
        panns_result = compute_panns_pairwise_diversity(
            gen_paths=all_audio_paths,
            panns_model=panns_model
        )
        panns_mean = panns_result[0] if panns_result is not None else None
        panns_std = panns_result[1] if panns_result is not None else None

        # 4. Compute Conditional Vendi Score
        try:
            evaluator = ConditionalEvaluation(sigma=0.5)
            cond_ent_results = evaluator.conditional_entropy(X=audio_embeds, Y=text_embeds, order=1)
            cond_entropy = cond_ent_results[0]
            cond_vendi_score = torch.exp(cond_entropy).item()
        except Exception as e:
            print(f"[Warning] Conditional Vendi Score failed for {folder_name}: {e}")
            cond_vendi_score = None

        # Write per-folder results to TXT
        with open(TXT_OUTPUT, "a", encoding="utf-8") as f:
            f.write(f"\n--- Results for {folder_name} (Valid files: {len(all_audio_paths)}) ---\n")
            f.write(f" CLAP Score      : {clap_score:.4f}\n")
            f.write(f" CLAP-FAD        : {fad_score:.4f}\n" if fad_score is not None else " CLAP-FAD        : FAILED\n")
            f.write(f" PANNs Diversity : {panns_mean:.4f} ± {panns_std:.4f}\n" if panns_mean is not None else " PANNs Diversity : FAILED\n")
            f.write(f" Cond-Vendi      : {cond_vendi_score:.4f}\n" if cond_vendi_score is not None else " Cond-Vendi      : FAILED\n")

        # Write per-folder row to CSV
        with open(CSV_OUTPUT, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow({
                "folder": folder_name,
                "count": len(all_audio_paths),
                "clap_score": f"{clap_score:.4f}" if clap_score is not None else "N/A",
                "clap_fad": f"{fad_score:.4f}" if fad_score is not None else "N/A",
                "panns_div_mean": f"{panns_mean:.4f}" if panns_mean is not None else "N/A",
                "panns_div_std": f"{panns_std:.4f}" if panns_std is not None else "N/A",
                "cond_vendi": f"{cond_vendi_score:.4f}" if cond_vendi_score is not None else "N/A",
            })

        results_summary.append({
            "folder": folder_name,
            "count": len(all_audio_paths),
            "clap": clap_score,
            "fad": fad_score,
            "panns_mean": panns_mean,
            "panns_std": panns_std,
            "cond_vendi": cond_vendi_score
        })

    # ==============================================================================
    # 5. Write Final Grand Summary Table to TXT & Mean/Std Row to CSV
    # ==============================================================================
    with open(TXT_OUTPUT, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 105 + "\n")
        f.write("                                     FINAL EVALUATION SUMMARY\n")
        f.write("=" * 105 + "\n")
        header = f"{'Folder Name':<50} | {'Count':<5} | {'CLAP ↑':<8} | {'CLAP-FAD ↓':<10} | {'PANNs Div ↑':<15} | {'CondVendi ↑':<12}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for res in results_summary:
            folder_str = res['folder'][:50]
            cnt_str = str(res['count'])
            clap_str = f"{res['clap']:.4f}" if res['clap'] is not None else "N/A"
            fad_str = f"{res['fad']:.4f}" if res['fad'] is not None else "N/A"
            panns_str = f"{res['panns_mean']:.4f}±{res['panns_std']:.3f}" if res['panns_mean'] is not None else "N/A"
            cond_vendi_str = f"{res['cond_vendi']:.4f}" if res['cond_vendi'] is not None else "N/A"

            f.write(f"{folder_str:<50} | {cnt_str:<5} | {clap_str:<8} | {fad_str:<10} | {panns_str:<15} | {cond_vendi_str:<12}\n")
        f.write("=" * 105 + "\n")

    if results_summary:
        def mean_std_str(vals):
            clean = [v for v in vals if v is not None]
            if not clean:
                return "N/A"
            std = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
            return f"{float(np.mean(clean)):.4f} ± {std:.4f}"

        with open(CSV_OUTPUT, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow({
                "folder": "MEAN +/- STD (ALL FOLDERS)",
                "count": sum(r["count"] for r in results_summary),
                "clap_score": mean_std_str([r["clap"] for r in results_summary]),
                "clap_fad": mean_std_str([r["fad"] for r in results_summary]),
                "panns_div_mean": mean_std_str([r["panns_mean"] for r in results_summary]),
                "panns_div_std": "-",
                "cond_vendi": mean_std_str([r["cond_vendi"] for r in results_summary]),
            })

    print(f"\nDone! Results saved to '{TXT_OUTPUT}' and '{CSV_OUTPUT}'.")
