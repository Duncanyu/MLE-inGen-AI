import src.asr.recorder as recorder
import src.asr.transcriber as transcriber

def get_prompt(model_size = "base", device = "cpu"):
    audio_path = recorder.record(sample_rate=16000)

    text = transcriber.transcribe(audio_path, model_size=model_size, device=device)

    return text