from nemo.collections.asr.models import EncDecMultiTaskModel
from lhotse import AudioSource
from lhotse.cut import MonoCut
from lhotse.cut import CutSet

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
    # Load Canary model
    canary_model = EncDecMultiTaskModel.from_pretrained(model_name)

    # Update decode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    # Load audio and convert to MonoCut
    audio = AudioSource.from_file(audio_file)
    cut = MonoCut(id="cut", start=0, duration=len(audio) / 16000, channel=0, recording=audio)

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
    # Load Canary model
    canary_model = EncDecMultiTaskModel.from_pretrained(model_name)

    # Update decode params
    decode_cfg = canary_model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    canary_model.change_decoding_strategy(decode_cfg)

    # Load audio and convert to MonoCut
    audio = AudioSource.from_file(audio_file)
    cut = MonoCut(id="cut", start=0, duration=len(audio) / 16000, channel=0, recording=audio)

    # Create a CutSet
    cut_set = CutSet.from_cuts([cut])
    predicted_text = canary_model.transcribe(
        cut_set,
        batch_size=batch_size,  # Batched inference
    )
    return predicted_text, language
