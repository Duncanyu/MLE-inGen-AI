import sounddevice as sd
import numpy as np
import tempfile
import wave
import webrtcvad

def save_wav(audio_data, sample_rate=16000):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    with wave.open(temp_file.name, 'wb') as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(sample_rate)
        wavf.writeframes(audio_data.tobytes())

    return temp_file.name


def record(sample_rate = 16000, frame_duration = 20, padding_duration = 0.4, vad_aggressiveness = 1):
    vad = webrtcvad.Vad(vad_aggressiveness)
    frame_size = int(sample_rate * frame_duration / 1000)
    num_padding_frames = int(padding_duration * 1000 / frame_duration)

    triggered = False
    buffer = []
    silent_frames = 0

    print("Listening... (stop speaking to end)")

    def callback(indata, frames, time, status):
        nonlocal triggered, silent_frames

        frame = bytes(indata)
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            if is_speech:
                triggered = True
                buffer.append(frame)
                silent_frames = 0
        else:
            buffer.append(frame)
            if is_speech:
                silent_frames = 0
            else:
                silent_frames += 1
                if silent_frames > num_padding_frames:
                    sd.stop()

    with sd.RawInputStream(
        samplerate = sample_rate,
        blocksize = frame_size,
        dtype = 'int16',
        channels = 1,
        callback = callback
    ):
        sd.sleep(10000)

    audio_data = np.frombuffer(b''.join(buffer), dtype='int16')
    print("Finished recording")

    return save_wav(audio_data, sample_rate)