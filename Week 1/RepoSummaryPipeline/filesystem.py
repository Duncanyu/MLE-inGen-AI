import json
import os

def save(data, filename, SAVE_DIR = "saved/"):
    path = os.path.join(SAVE_DIR, filename)
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load(filename, SAVE_DIR = "saved/"):
    path = os.path.join(SAVE_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
