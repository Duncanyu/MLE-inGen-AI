import os
import sys
import shutil
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE
DATA = BASE / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
INDEX = DATA / "index"

CHUNKS_FILE = INTERIM / "chunks.jsonl"
FAISS_FILE = INDEX / "faiss.index"
META_PKL = INDEX / "meta.pkl"

OPENAI_KEY = None
try:
    from keys import OPENAI_KEY as _K
    OPENAI_KEY = _K
except Exception:
    try:
        from keys import openai_key as _k
        OPENAI_KEY = _k
    except Exception:
        OPENAI_KEY = None

def set_env():
    env = os.environ.copy()
    if OPENAI_KEY and OPENAI_KEY.strip():
        env["OPENAI_API_KEY"] = OPENAI_KEY
    return env

def run_py(script: str, env=None):
    script_path = SRC / script
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    print(f"\nRunning {script} …")
    subprocess.run([sys.executable, str(script_path)], check=True, env=env or os.environ.copy())
    print(f"{script} finished")

def ensure_dirs():
    INTERIM.mkdir(parents=True, exist_ok=True)
    INDEX.mkdir(parents=True, exist_ok=True)

def clean_outputs():
    if INTERIM.exists():
        shutil.rmtree(INTERIM)
    if INDEX.exists():
        shutil.rmtree(INDEX)
    ensure_dirs()
    print("cleaned data/interim and data/index")

def check_file(path: Path, hint: str):
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"expected but missing: {path}\nghint: {hint}")
    print(f" -> found {path}")

def step_chunk(env):
    ensure_dirs()
    if not RAW.exists():
        raise FileNotFoundError(f"put your files here: {RAW}")
    run_py("chunking.py", env=env)
    check_file(CHUNKS_FILE, "did chunking finish without errors?")

def step_index(env):
    ensure_dirs()
    check_file(CHUNKS_FILE, "run chunking first!")
    run_py("faiss_script.py", env=env)
    check_file(FAISS_FILE, "faiss_script.py should produce faiss.index")
    check_file(META_PKL, "faiss_script.py should produce meta.pkl")

def step_query(env):
    check_file(CHUNKS_FILE, "run chunking first!")
    if not (OPENAI_KEY and OPENAI_KEY.strip()):
        print("no OPENAI key detected in keys.py the Query model may fail. "
              "put in an OPENAI_KEY or openai_key to use chatgpt")
    run_py("query.py", env=env)

def main():
    env = set_env()
    step_chunk(env)
    step_index(env)
    print("\npipeline done.")
    try:
        step_query(env)
    except FileNotFoundError as e:
        print(str(e))

if __name__ == "__main__":
    main()