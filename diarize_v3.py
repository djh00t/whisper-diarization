import warnings

# Suppress the specific UserWarning
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")

import argparse
import os
import torch
import torchaudio
from transcription_models.canary_model_v4 import load_canary_model, transcribe_batched
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
import logging
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

args = parser.parse_args()

model_name = args.transcription_model.lower()
print("#" * 79)
print(f"DEBUG: Transcription Model: {model_name}")
print("#" * 79)

# Model Selection - Check if selected model is Whisper or Canary
if model_name.startswith("whisper-"):
    from transcription_models.whisper_model import transcribe_batched as whisper_transcribe_batched, WHISPER_MODELS
    model = None
    model_name = WHISPER_MODELS[model_name]
    print("#" * 79)
    print(f"DEBUG: Model Name: {model_name}")
elif model_name.startswith("canary-"):
    from transcription_models.canary_model_v4 import transcribe_batched as canary_transcribe_batched, CANARY_MODELS
    model = load_canary_model(CANARY_MODELS[model_name], args.device)
    print("#" * 79)
    print(f"DEBUG: Model Name: {model_name}")
else:
    raise ValueError(f"Unsupported transcription model: {args.transcription_model}")

if args.stemming:
    # Isolate vocals from the rest of the audio
    return_code = os.system(
        f'python -m demucs.separate -n htdemucs --two-stems=vocals "{args.audio}" -o "temp_outputs"'
    )

    if return_code != 0:
        logging.warning(
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

# Transcribe the audio file
if model_name.startswith("whisper-"):
    transcribe_results, language, audio_waveform = whisper_transcribe_batched(
        audio_file=vocal_target,
        language=args.language,
        batch_size=args.batch_size,
        model_name=model_name,
        compute_dtype=mtypes[args.device],
        suppress_numerals=args.suppress_numerals,
        device=args.device,
    )
else:
    transcribe_results, language, audio_waveform = canary_transcribe_batched(
        model=model,
        audio_file=vocal_target,
        language=args.language,
        batch_size=args.batch_size,
        compute_dtype=mtypes[args.device],
        suppress_numerals=args.suppress_numerals,
        device=args.device,
    )

# Handle language detection failure
if language is None:
    language = "en"  # Set a default language if detection fails
    print("WARNING: Language detection failed, defaulting to English (en)")

# Forced Alignment
alignment_model, alignment_tokenizer, alignment_dictionary = load_alignment_model(
    args.device,
    dtype=torch.float16 if args.device == "cuda" else torch.float32,
)

# Ensure audio waveform is a torch tensor
audio_waveform = torch.tensor(audio_waveform)
print(f"DEBUG: Audio Waveform Tensor Shape: {audio_waveform.shape}")

# Split audio waveform for forced alignment
max_len = 5000  # Adjust segment length to avoid tensor size issues
audio_waveform_segments = torch.split(audio_waveform, max_len, dim=1)
print(f"DEBUG: Number of Segments: {len(audio_waveform_segments)}")

emissions = []
stride = None

for idx, segment in enumerate(audio_waveform_segments):
    print(f"DEBUG: Processing Segment {idx + 1}/{len(audio_waveform_segments)} with Shape: {segment.shape}")
    try:
        segment_emissions, segment_stride = generate_emissions(
            alignment_model, segment, batch_size=args.batch_size
        )
        print(f"DEBUG: Segment Emissions Shape: {segment_emissions.shape}, Stride: {segment_stride}")
        emissions.append(segment_emissions)
        stride = segment_stride
    except RuntimeError as e:
        print(f"ERROR: Processing Segment {idx + 1}/{len(audio_waveform_segments)} failed with error: {e}")
        break

# Verify emissions before concatenation
if emissions:
    try:
        emissions = torch.cat(emissions, dim=1)
        print(f"DEBUG: Combined Emissions Shape: {emissions.shape}")
    except Exception as e:
        print(f"ERROR: Concatenating Emissions failed with error: {e}")

if not isinstance(emissions, torch.Tensor):
    print("ERROR: Emissions is not a tensor. Exiting.")
    exit(1)

del alignment_model
torch.cuda.empty_cache()

full_transcript = "".join(segment["text"] for segment in transcribe_results)

tokens_starred, text_starred = preprocess_text(
    full_transcript,
    romanize=True,
    language=langs_to_iso[language],
)

segments, blank_id = get_alignments(
    emissions,
    tokens_starred,
    alignment_dictionary,
)

spans = get_spans(tokens_starred, segments, alignment_tokenizer.decode(blank_id))

word_timestamps = postprocess_results(text_starred, spans, stride)

# Convert audio to mono for NeMo compatibility
ROOT = os.getcwd()
temp_path = os.path.join(ROOT, "temp_outputs")
os.makedirs(temp_path, exist_ok=True)
torchaudio.save(
    os.path.join(temp_path, "mono_file.wav"),
    audio_waveform.unsqueeze(0).float(),
    16000,
    channels_first=True,
)

# Initialize NeMo MSDD diarization model
msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(args.device)
msdd_model.diarize()

del msdd_model
torch.cuda.empty_cache()

# Reading timestamps <> Speaker Labels mapping
speaker_ts = []
with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
    lines = f.readlines()
    for line in lines:
        line_list = line.split(" ")
        s = int(float(line_list[5]) * 1000)
        e = s + int(float(line_list[8]) * 1000)
        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

if language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences
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
    logging.warning(
        f"Punctuation restoration is not available for {language} language. Using the original punctuation."
    )

wsm = get_realigned_ws_mapping_with_punctuation(wsm)
ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

with open(f"{os.path.splitext(args.audio)[0]}.txt", "w", encoding="utf-8-sig") as f:
    get_speaker_aware_transcript(ssm, f)

with open(f"{os.path.splitext(args.audio)[0]}.srt", "w", encoding="utf-8-sig") as srt:
    write_srt(ssm, srt)

cleanup(temp_path)
