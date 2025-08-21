import subprocess
import sys
import os

SCRIPTS = [
    ("preparation", "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/prep.py"),
    ("lora", "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/train_lora.py"),
    # ("full", "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/train_full.py"),
    ("eval", "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/eval.py"),
    ("Report creating", "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/eval.py"),
]

def run_step(name, path):
    print(f"\n=== running: {name} ({path}) ===\n")
    result = subprocess.run([sys.executable, path])
    if result.returncode != 0:
        print(f"failed: {name}")
        sys.exit(result.returncode)

def main():
    for name, path in SCRIPTS:
        if not os.path.exists(path):
            print(f"missing script: {path}")
            sys.exit(1)
        run_step(name, path)
    print("\ncomp[leted! :)\n")

if __name__ == "__main__":
    main()
