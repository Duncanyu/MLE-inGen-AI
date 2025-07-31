import yt_dlp
import whisper
import json
import os

urls = [
    'https://www.youtube.com/watch?v=8zKVrUtW-Oc',
    'https://www.youtube.com/watch?v=068nfPdtssI',
    'https://www.youtube.com/watch?v=lfJOlp2sN18',
    'https://www.youtube.com/watch?v=RBqf25pbVZ0',
    'https://www.youtube.com/watch?v=5d1e8V-RsxQ',
    'https://www.youtube.com/watch?v=8Soq-UzFxPo&pp=0gcJCccJAYcqIYzv',
    'https://www.youtube.com/watch?v=SQM9qjuELws',
    'https://www.youtube.com/watch?v=rjAsY4P87SI',
    'https://www.youtube.com/watch?v=F6G0e-cxDPk',
    'https://www.youtube.com/watch?v=TzJS9bUNi2U'
]

download_folder = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Bonus3-ASR/yt_dlp-out"
jsonl_path = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Bonus3-ASR/talks_transcripts.jsonl"

ydl_opts = {
    'format': 'm4a/bestaudio/best',
    'outtmpl': f'{download_folder}/%(id)s.%(ext)s',
    'noplaylist': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(urls)

model = whisper.load_model("base")

with open(jsonl_path, "w", encoding="utf-8") as out_f:
    for url in urls:
        video_id = url.split("v=")[-1].split("&")[0]
        audio_path = f"{download_folder}/{video_id}.m4a"

        if not os.path.exists(audio_path):
            print(f"not found: {audio_path}")
            continue
        else:
            print(f" == transcribing: {audio_path}")

        try:
            result = model.transcribe(audio_path)
        except Exception as e:
            print(f"not work {audio_path}: {e}")
            continue

        for segment in result["segments"]:
            json_line = {
                "video": url,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
            out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

print(f"saved to {jsonl_path}")
