from nemo.collections.asr.models import EncDecMultiTaskModel
import torchaudio
from langdetect import detect
import numpy as np

# List of available Canary models
CANARY_MODELS = {
    "canary-1b": "nvidia/canary-1b",
}

def transcribe(
    audio_file: str,
    language: str,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):

    print("################################################################")
    print(f"Loading Canary Model: {model_name}")
    print("################################################################")
    # Load Canary model
    canary_model = EncDecMultiTaskModel.from_pretrained(model_name)

    # Update decode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    # Load audio and convert to MonoCut
    waveform, sample_rate = torchaudio.load(audio_file)
    audio = MonoCut(id="cut", start=0, duration=waveform.shape[1] / sample_rate, channel=0, recording=waveform)
    cut = MonoCut(id="cut", start=0, duration=waveform.shape[1] / sample_rate, channel=0, recording=audio)

    # Create a CutSet
    cut_set = CutSet.from_cuts([cut])
    predicted_text = canary_model.transcribe(
        paths2audio_files=[audio_file],
        batch_size=1,  # Non-batched inference
    )
    return predicted_text, language

def transcribe_batched(
    audio_file: str,
    language: str,
    batch_size: int,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):

    print("################################################################")
    print(f"Loading Canary Model: {model_name}")
    print("################################################################")
    # Load Canary model
    canary_model = EncDecMultiTaskModel.from_pretrained(model_name)

    # Update decode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    # Load the audio waveform
    waveform, sample_rate = torchaudio.load(audio_file)
    audio_waveform = waveform.squeeze(0).numpy()

    # Ensure audio_waveform is in the same format as whisper_model.py
    audio_waveform = np.ascontiguousarray(audio_waveform)

    # Transcribe the audio
    predicted_text = canary_model.transcribe(
        audio=audio_file,  # Provide the list of audio files
        batch_size=batch_size if batch_size > 0 else 1,  # Use batch size or non-batched inference
        return_hypotheses=False,  # Return text instead of hypotheses
        source_lang='en',  # Source language
        target_lang='en',  # Target language
        num_workers=0,  # Number of workers for DataLoader
        verbose=True,  # Display tqdm progress bar
    )

    print("################################################################")
    print(f"predicted_text: {predicted_text}")
    print("################################################################")
    
    # Detect the language from the transcriptions
    detected_language = detect(" ".join(predicted_text))
    print("################################################################")
    print(f"detected_language: {detected_language}")
    print("################################################################")

    # Prepare whisper_results in the expected format
    whisper_results = [{'text': t} for t in predicted_text]
    print("################################################################")
    print(f"whisper_results: {whisper_results}")
    print("################################################################")

    return whisper_results, detected_language, audio_waveform