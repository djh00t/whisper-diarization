import warnings
import logging
import sys

warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")

import argparse
import os
import torch
import torchaudio
from transcription_models.canary_model_v4 import load_canary_model, transcribe_batched as canary_transcribe_batched
from helpers import (
    whisper_langs,
    langs_to_iso,
    punct_model_langs,
    create_config,
    get_words_speaker_mapping,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    write_srt,
    cleanup,
)
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# Define mtypes
mtypes = {
    "cuda": "float16",
    "cpu": "float32"
}

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-a", "--audio", help="name of the target audio file", required=True
)
parser.add_argument(
    "--no-stem",
    action="store_false",
    dest="stemming",
    default=True,
    help="Disables source separation. This helps with long files that don't contain a lot of music.",
)
parser.add_argument(
    "--suppress_numerals",
    action="store_true",
    dest="suppress_numerals",
    default=False,
    help="Suppresses Numerical Digits. This helps the diarization accuracy but converts all digits into written text.",
)
parser.add_argument(
    "--batch-size",
    type=int,
    dest="batch_size",
    default=8,
    help="Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference",
)
parser.add_argument(
    "--language",
    type=str,
    default=None,
    choices=whisper_langs,
    help="Language spoken in the audio, specify None to perform language detection",
)
parser.add_argument(
    "--device",
    dest="device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="if you have a GPU use 'cuda', otherwise 'cpu'",
)
parser.add_argument(
    "--transcription-model",
    type=str,
    default="whisper-small",
    help="Choose the transcription model to use. Available models: whisper-tiny, whisper-tiny.en, whisper-base, whisper-base.en, whisper-small, whisper-small.en, whisper-medium, whisper-medium.en, whisper-large, whisper-large-v2, whisper-large-v3, canary-1b",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug logging"
)

args = parser.parse_args()

# Configure logger
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if args.debug:
    logger.debug("Debug logging enabled")

# Initialize language variable
language = args.language

# Handle language detection failure
if language is None:
    language = "en"  # Set a default language if detection fails
    logger.warning("Language detection failed, defaulting to English (en)")

if args.stemming:
    # Isolate vocals from the rest of the audio
    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
    )

    if return_code != 0:
        logger.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = args.audio
    else:
        vocal_target = os.path.join(
            "temp_outputs",
            "htdemucs",
            os.path.splitext(os.path.basename(args.audio))[0],
            "vocals.wav",
        )
else:
    vocal_target = args.audio

# Get model name from arguments
model_name = args.transcription_model.lower()
logger.info(f" Selected Transcription Model: {model_name}")

# Model Selection - Check if selected model is Whisper or Canary
if model_name.startswith("whisper-"):
    # Import Whisper Model
    logger.info(" Loading Whisper Model")
    from transcription_models.whisper_model import transcribe_batched as whisper_transcribe_batched, WHISPER_MODELS
    model = None
    model_name = WHISPER_MODELS[model_name]

    # Transcribe the audio file
    logger.info(" Transcribing Audio File")
    transcribe_results, language, audio_waveform = whisper_transcribe_batched(
        audio_file=vocal_target,
        language=args.language,
        batch_size=args.batch_size,
        model_name=model_name,
        compute_dtype=mtypes[args.device],
        suppress_numerals=args.suppress_numerals,
        device=args.device,
    )
    logger.info(" Transcription Complete")

elif model_name.startswith("canary-"):
    # Import Canary Model
    logger.info(" Loading Canary Model")
    from transcription_models.canary_model_v4 import transcribe_batched as canary_transcribe_batched, CANARY_MODELS
    model = load_canary_model(CANARY_MODELS[model_name], args.device)

    # Transcribe the audio file
    logger.info(" Transcribing Audio File")
    transcribe_results, language, audio_waveform = canary_transcribe_batched(
        model=model,
        audio_file=vocal_target,
        language=args.language,
        batch_size=args.batch_size,
        compute_dtype=mtypes[args.device],
        suppress_numerals=args.suppress_numerals,
        device=args.device,
    )
    logger.info(" Transcription Complete")

else:
    raise ValueError(f"Unsupported transcription model: {args.transcription_model}")

# Forced Alignment
logger.info(" Loading Forced Alignment Model")
alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

# Ensure audio waveform is a torch tensor and on the correct device with the right dtype
logger.info(" Converting Audio Waveform to Tensor")
audio_waveform = torch.tensor(audio_waveform).to(args.device).to(alignment_model.dtype)
if audio_waveform.dim() == 1:
    audio_waveform = audio_waveform.unsqueeze(0)  # Add batch dimension if not present
logger.debug(f"Audio Waveform Tensor Shape: {audio_waveform.shape}")

# Split audio waveform for forced alignment
window_size = 30 * 16000  # 30 seconds window size in samples
context_size = 2 * 16000  # 2 seconds context size in samples
logger.info(" Splitting Audio Waveform for Forced Alignment")
audio_waveform_segments = torch.split(audio_waveform, window_size - context_size, dim=1)
logger.debug(f"Number of Segments: {len(audio_waveform_segments)}")

# Setup variables for alignment
emissions = []
stride = None

# Perform forced alignment on each segment
logger.info(" Performing Forced Alignment on Audio Segments")
for idx, segment in enumerate(audio_waveform_segments):
    logger.debug(f"Processing Segment {idx + 1}/{len(audio_waveform_segments)} with Shape: {segment.shape}")
    try:
        padded_waveform = segment.squeeze(0)  # Remove batch dimension
        logger.debug(f"Padded Waveform Shape: {padded_waveform.shape}")
        
        segment_emissions, segment_stride = generate_emissions(
            alignment_model, padded_waveform, window_size // 16000, context_size // 16000, args.batch_size
        )
        logger.debug(f"Segment Emissions Shape: {segment_emissions.shape}, Stride: {segment_stride}")
        emissions.append(segment_emissions)
        stride = segment_stride
        logger.info(f" Segment {idx + 1}/{len(audio_waveform_segments)} Alignment Complete")
    except RuntimeError as e:
        logger.error(f" Processing Segment {idx + 1}/{len(audio_waveform_segments)} failed with error: {e}")
        break

# Verify emissions before concatenation
logger.info(" Verifying Emissions before Concatenation")
if emissions:
    try:
        # Find the maximum length of emissions
        max_len = max(e.size(0) for e in emissions)
        # Pad emissions to the maximum length
        padded_emissions = [torch.nn.functional.pad(e, (0, 0, 0, max_len - e.size(0))) for e in emissions]
        emissions = torch.cat(padded_emissions, dim=1)
        logger.debug(f"Combined Emissions Shape: {emissions.shape}")
        logger.info(" Concatenating Emissions Complete")
    except Exception as e:
        logger.error(f"Concatenating Emissions failed with error: {e}")

if not isinstance(emissions, torch.Tensor):
    logger.error("Emissions is not a tensor. Exiting.")
    exit(1)

# Cleanup alignment model
logger.info(" Cleaning up Alignment Model")
del alignment_model
logger.info(" Freeing up GPU Memory")
torch.cuda.empty_cache()

# Preprocess text for alignment
logger.info(" Preprocessing Text for Alignment")
full_transcript = "".join(segment["text"] for segment in transcribe_results)

# Perform forced alignment
logger.info(" Performing Forced Alignment")
tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[language],
)

# Get alignments
logger.info(" Getting Alignments")
segments, blank_id = get_alignments(
    emissions,
    tokens_starred,
    alignment_dictionary,
)

# Get spans
logger.info(" Getting Spans")
spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))

# Postprocess results
logger.info(" Postprocessing Results")
word_timestamps = postprocess_results(text_starred, spans, stride)

# Convert audio to mono for NeMo compatibility
logger.info(" Preprocessing Audio to Mono for NeMo Diarization & Compatibility")
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
logger.debug(f"Temporary Path: {temp_path}")
os.makedirs(temp_path, exist_ok=True)
logger.debug(f"Temporary Directory Created: {temp_path}")
logger.info(
    " Normalizing Audio as:\n"
    "\t - Channels:    1\n"
    "\t - Depth:       16-bit\n"
    "\t - Modulation:  PCM\n"
    "\t - Sample Rate: 16 kHz\n"
    "\t - File Format: WAV"
)
# Move audio waveform to CPU and ensure it is 2D
audio_waveform_cpu = audio_waveform.cpu().float()
logger.debug(f"Audio Waveform Tensor Shape before ensuring 2D: {audio_waveform_cpu.shape}")
if audio_waveform_cpu.dim() == 1:
    audio_waveform_cpu = audio_waveform_cpu.unsqueeze(0)
elif audio_waveform_cpu.dim() == 2:
    pass
else:
    raise RuntimeError("Audio waveform must be a 1D or 2D tensor.")
logger.debug(f"Audio Waveform Tensor Shape for saving: {audio_waveform_cpu.shape}")

torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    audio_waveform_cpu,
    16000,
    channels_first=True,
)
logger.info(f" Normalized file saved to: {os.path.join(temp_path, 'mono_file.wav')}")

# Initialize NeMo MSDD diarization model
logger.info(" Initializing NeMo MSDD Diarization Model")
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)

# Perform Speaker Diarization
logger.info(" Performing Speaker Diarization")
msdd_model.diarize()

# Cleanup MSDD Model
logger.info(" Cleaning up MSDD Model")
del msdd_model

# Free up GPU Memory
logger.info(" Freeing up GPU Memory")
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping
# Setup speaker timestamps
logger.info(" Setting up Speaker Timestamps")

logger.info(" Reading Speaker Timestamps")
speaker_ts = []

logger.info(" Reading RTTM File")
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:

    # Read lines from the RTTM file
    lines = f.readlines()

    # Create line_number variable for tracking
    line_number = 0

    # Get start time, end time, and speaker label for each line
    for line in lines:
        # Add 1 to line_number
        line_number += 1

        # Log processing line number
        logger.info(f" Processing Line {line_number}/{len(lines)}")
        
        # Split the line by space
        line_list = line.split(" ")

        # Extract start and end times
        s = int(float(line_list[5]) * 1000)

        # Calculate end time
        e = s + int(float(line_list[8]) * 1000)

        # Extract speaker label
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        # Log speaker, start, and end times for line_number
        logger.info(f" Speaker: {int(line_list[11].split('_')[-1])}, Start: {s}, End: {e}")

# Get words-speaker mapping
logger.info(" Getting Words-Speaker Mapping")
wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
    logger.info(" Restoring Punctuation")
    punct_model = PunctuationModel(model="kredor/punctuate-all")

    words_list = list(map(lambda x: x["word"], wsm))

    labled_words = punct_model.predict(words_list, chunk_size=230)

    ending_puncts = ".?!"
    model_puncts = ".,;:!?"

    # We don't want to punctuate U.S.A. with a period. Right?
    is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

    for word_dict, labeled_tuple in zip(wsm, labled_words):
        word = word_dict["word"]
        if (
            word
            and labeled_tuple[1] in ending_puncts
            and (word[-1] not in model_puncts or is_acronym(word))
        ):
            word += labeled_tuple[1]
            if word.endswith(".."):
                word = word.rstrip(".")
            word_dict["word"] = word

else:
    logger.warning(
        f"Punctuation restoration is not available for {language} language. Using the original punctuation."
    )

# Get realigned words-speaker mapping with punctuation
logger.info(" Getting Realigned Words-Speaker Mapping with Punctuation")
wsm = get_realigned_ws_mapping_with_punctuation(wsm)

# Get sentences-speaker mapping
logger.info(" Getting Sentences-Speaker Mapping")
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

# Write pre-diarization transcript to text file
logger.info(" Writing Pre-Diarization Transcript to Text File")
with open(f"{os.path.splitext(args.audio)[0]}_pre_diarization.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f, pre_diarization=True)

# Write post-diarization speaker-aware transcript to text file
logger.info(" Writing Post-Diarization Speaker-Aware Transcript to Text File")
with open(f"{os.path.splitext(args.audio)[0]}_post_diarization.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f, pre_diarization=False)

# Write SRT with speaker attribution
logger.info(" Writing Speaker-Aware Transcript to SRT File")
with open(f"{os.path.splitext(args.audio)[0]}_speaker.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt, include_speaker=True)

# Write SRT without speaker attribution
logger.info(" Writing Transcript to SRT File without Speaker Attribution")
with open(f"{os.path.splitext(args.audio)[0]}_no_speaker.srt", "w", encoding="utf-8-sig") as srt_no_speaker:
    write_srt(ssm, srt_no_speaker, include_speaker=False)

# Cleanup temporary files
logger.info(" Cleaning up Temporary Files")
cleanup(temp_path)

# Log completion
logger.info(" Processing Complete")
