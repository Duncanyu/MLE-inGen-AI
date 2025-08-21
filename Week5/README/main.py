import sys, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("preparation", "prep.py"),
    ("lora",       "train_lora.py"),
    # (full-tune",      "train_full.py"),
    ("eval",   "eval.py"),
    ("reporting",     "report.py"),
]

def run_step(name, rel):
    path = BASE_DIR / rel
    if not path.exists():
        print(f"missing script: {path}")
        sys.exit(1)
    print(f"\n=== running: {name} ({path}) ===\n")
    result = subprocess.run([sys.executable, str(path)], cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"failed: {name}")
        sys.exit(result.returncode)

def main():
    for name, rel in SCRIPTS:
        run_step(name, rel)
    print("\ncompleted :)")

if __name__ == "__main__":
    main()
