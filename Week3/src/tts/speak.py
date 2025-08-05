from TTS.api import TTS
import tempfile
import shutil
import subprocess
import sys

tts = None

def generate_tts(text: str, model: str, speaker: str):
    print("Generating TTS...")
    global tts
    
    if tts is None:
        tts = TTS(model)
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()

    tts.tts_to_file(text = text, file_path = temp_file_path, speaker = speaker)
    return temp_file_path
