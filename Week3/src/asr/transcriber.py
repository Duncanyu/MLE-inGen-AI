from faster_whisper import WhisperModel

def transcribe(audio_path, model_size = "small", device = "cpu"):
    print("Initializing Whisper...")
    model = WhisperModel(model_size, device = device, compute_type = "int8")

    print(f"Transcribing...")
    segments, info = model.transcribe(audio_path)
    text = " ".join([segment.text for segment in segments])

    return text