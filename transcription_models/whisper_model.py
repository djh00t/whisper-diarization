from faster_whisper import WhisperModel
from helpers import find_numeral_symbol_tokens, wav2vec2_langs

import torch
import whisperx

# List of available Whisper models
WHISPER_MODELS = {
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-tiny.en": "openai/whisper-tiny.en",
    "whisper-base": "openai/whisper-base",
    "whisper-base.en": "openai/whisper-base.en",
    "whisper-small": "openai/whisper-small",
    "whisper-small.en": "openai/whisper-small.en",
    "whisper-medium": "openai/whisper-medium",
    "whisper-medium.en": "openai/whisper-medium.en",
    "whisper-large": "openai/whisper-large",
    "whisper-large-v2": "openai/whisper-large-v2",
    "whisper-large-v3": "openai/whisper-large-v3",
}



def transcribe(
    audio_file: str,
    language: str,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    # Faster Whisper non-batched
    whisper_model = WhisperModel(model_name, device=device, compute_type=compute_dtype)

    if suppress_numerals:
        numeral_symbol_tokens = find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
    else:
        numeral_symbol_tokens = None

    if language is not None and language in wav2vec2_langs:
        word_timestamps = False
    else:
        word_timestamps = True

    segments, info = whisper_model.transcribe(
        audio_file,
        language=language,
        beam_size=5,
        word_timestamps=word_timestamps,
        suppress_tokens=numeral_symbol_tokens,
        vad_filter=True,
    )
    whisper_results = [segment._asdict() for segment in segments]
    del whisper_model
    torch.cuda.empty_cache()
    return whisper_results, info.language

def transcribe_batched(
    audio_file: str,
    language: str,
    batch_size: int,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
    logger,
):

    logger.info(f" Loading Whisper Model: {model_name}")
    
    # Remove "whisper-" prefix from model_name if present
    if model_name.startswith("openai/whisper-"):
        model_name = model_name[len("openai/whisper-"):]

    # Load the Whisper model using whisperx library
    whisper_model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_dtype,
        asr_options={"suppress_numerals": suppress_numerals},
    )

    # Load the audio waveform and ensure it is mono
    logger.info(f" Loading Audio File: {audio_file}")
    audio = whisperx.load_audio(audio_file)
    if audio.shape[0] > 1:
        logger.info(" Converting Audio to Mono")
        audio = audio.mean(dim=0, keepdim=True)

    # Transcribe the audio
    logger.info(f" Transcribing Audio: {audio_file}")
    result = whisper_model.transcribe(audio, language=language, batch_size=batch_size)

    # Delete the model and free up memory
    logger.info(" Deleting Whisper Model")
    del whisper_model

    # Clear the GPU memory
    logger.info(" Clearing GPU Memory")
    torch.cuda.empty_cache()

    # Print debug information
    logger.debug(f"Transcription Results: {result['segments']}")
    logger.debug(f"Language: {result['language']}")
    logger.debug(f"Audio: {audio}")

    # Return the transcribed segments, language, and audio
    return result["segments"], result["language"], audio
