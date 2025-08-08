import sounddevice as sd
import numpy as np
import tempfile
import wave
import webrtcvad
from collections import deque
import time

def save_wav(audio_data, sample_rate=16000):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(temp_file.name, 'wb') as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(sample_rate)
        wavf.writeframes(audio_data.tobytes())
    return temp_file.name

def record(
    sample_rate = 16000,
    frame_duration = 30,
    padding_duration = 0.5,
    vad_aggressiveness = 2,
    max_wait_for_speech = 3.0,
):
    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_size = int(sample_rate * frame_duration / 1000)
    num_padding_frames = int(padding_duration * 1000 / frame_duration)

    ring_buffer = deque(maxlen=num_padding_frames)
    voiced_frames = []
    all_frames = []

    triggered = False
    start_time = time.time()

    print("Listening... (speak to record, stop talking to end)")

    def callback(indata, frames, time_info, status):
        nonlocal triggered, ring_buffer, voiced_frames, all_frames

        frame = bytes(indata)
        is_speech = vad.is_speech(frame, sample_rate)

        all_frames.append(frame)

        current_time = time.time()

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                print("Speech detected, recording...", flush=True)
                voiced_frames.extend(f for f, _ in ring_buffer)
                ring_buffer.clear()
            elif current_time - start_time > max_wait_for_speech:
                print("No speech detected within time limit, stopping...", flush=True)
                sd.stop()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                print("Silence detected, stopping...", flush=True)
                sd.stop()

    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=frame_size,
        dtype='int16',
        channels=1,
        callback=callback
    ):
        buffer_time = int((max_wait_for_speech + silence_timeout + 2) * 1000)
        sd.sleep(buffer_time)

    if not voiced_frames:
        print("No speech detected.")
        audio_data = np.frombuffer(b''.join(all_frames), dtype='int16')
    else:
        audio_data = np.frombuffer(b''.join(voiced_frames), dtype='int16')

    print("Finished recording")
    return save_wav(audio_data, sample_rate)
