import torch
from nemo.collections.asr.models import EncDecMultiTaskModel
import torchaudio
import os
import tempfile

# List of available Canary models
CANARY_MODELS = {
    "canary-1b": "nvidia/canary-1b",
}

def load_canary_model(model_name, device):
    print(f"DEBUG: Device:                      {device}")
    try:
        model = EncDecMultiTaskModel.from_pretrained(model_name)
    except Exception as e:
        raise ValueError(f"Failed to load model {model_name} from Hugging Face: {e}")

    model.to(device)
    model.eval()
    return model

def save_audio_to_tempfile(audio_data, sr, tmpfs_dir='/dev/shm'):
    temp_file_path = os.path.join(tmpfs_dir, next(tempfile._get_candidate_names()) + ".wav")
    torchaudio.save(temp_file_path, audio_data, sr)
    print(f"DEBUG: Saved Audio to Temp File:    {temp_file_path}")
    return temp_file_path

def estimate_timestamps(transcriptions, audio_duration):
    num_segments = len(transcriptions)
    segment_duration = audio_duration / num_segments
    timestamps = []
    current_time = 0.0

    for i in range(num_segments):
        start_time = current_time
        end_time = start_time + segment_duration
        timestamps.append((start_time, end_time))
        current_time = end_time

    return timestamps

def transcribe(
    audio_file: str,
    language: str,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):
    model = load_canary_model(model_name, device)

    # Load and preprocess the audio file
    audio_data, sr = torchaudio.load(audio_file)
    audio_duration = audio_data.shape[1] / sr  # Calculate audio duration in seconds
    if sr != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_data = transform(audio_data)

    # Save the audio data to a temporary file in tmpfs
    temp_audio_file = save_audio_to_tempfile(audio_data, 16000)

    # Perform transcription
    with torch.no_grad():
        transcription = model.transcribe(audio=[temp_audio_file])

    # Debug print to inspect transcription output
    print("#" * 79)
    print("DEBUG: Transcription Output: ", transcription)
    print("#" * 79)
    
    # Estimate timestamps
    timestamps = estimate_timestamps(transcription, audio_duration)

    # Post-process transcription to include timestamps
    transcriptions = []
    for i, segment in enumerate(transcription):
        transcriptions.append({
            'text': segment,
            'start': timestamps[i][0],
            'end': timestamps[i][1],
        })

    # Clean up and release resources
    del model
    os.remove(temp_audio_file)
    torch.cuda.empty_cache()

    # Assuming the transcription is returned in a compatible format
    return transcriptions, language, audio_data.numpy()

def transcribe_batched(
    audio_file: str,
    language: str,
    batch_size: int,
    model_name: str,
    compute_dtype: str,
    suppress_numerals: bool,
    device: str,
):

    print(f"DEBUG: Loading Model:               {model_name}")
    model = load_canary_model(model_name, device)

    # Load and preprocess the audio file
    print(f"DEBUG: Loading and Preprocessing    {audio_file}")
    audio_data, sr = torchaudio.load(audio_file)

    # Debug print to inspect audio data shape
    print(f"DEBUG: Audio Data Shape:            {audio_data.shape}")
    audio_duration = audio_data.shape[1] / sr  # Calculate audio duration in seconds

    # Resample audio data if necessary
    if sr != 16000:
        print("DEBUG: Resampling Audio Data to:    16kHz")
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_data = transform(audio_data)
    
    # Ensure batch size is valid
    batch_size = min(batch_size, audio_data.shape[1])

    # Split the audio data into batches
    print(f"DEBUG: Breaking into batches of:    {batch_size} samples")
    audio_batches = torch.split(audio_data, batch_size)

    # Perform transcription
    transcriptions = []
    batch_start_time = 0.0
    print("DEBUG: Transcribing Audio Batches")
    with torch.no_grad():
        for batch in audio_batches:
            print(f"DEBUG: Processing Batch:            {batch}")
            
            batch_duration = batch.shape[1] / sr
            print(f"DEBUG: Batch Duration:              {batch_duration} sec")

            temp_audio_file = save_audio_to_tempfile(batch, 16000)
            print(f"DEBUG: Transcribing Audio Batch:    {temp_audio_file}")

            transcription = model.transcribe(audio=[temp_audio_file])
            print(f"DEBUG: Transcription Output: \n{transcription}")

            # Generate timestamps for each segment in the batch
            segment_duration = batch_duration / len(transcription)
            print(f"DEBUG: Segment Duration:            {segment_duration} sec")
            
            for i, segment in enumerate(transcription):
                start_time = batch_start_time + i * segment_duration
                end_time = start_time + segment_duration
                transcriptions.append({
                    'text': segment,
                    'start': start_time,
                    'end': end_time,
                })
                
            batch_start_time += batch_duration
            os.remove(temp_audio_file)

    # Clean up and release resources
    del model
    torch.cuda.empty_cache()

    return transcriptions, language, audio_data.numpy()
