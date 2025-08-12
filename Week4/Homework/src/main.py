import os
import sys
import subprocess
from pathlib import Path

ARXIV_QUERY = "cat:cs.CL"
ARXIV_MAX_RESULTS = 200
SCRIPTS = ["scrape.py", "extract_clean.py", "chunking.py", "faiss_script.py"]

BASE = Path(__file__).resolve().parent

def run(script, env=None):
    print(f"\n == Running {script}... ==")
    subprocess.run([sys.executable, str(BASE / script)], check=True, env=env)

def main():
    env = os.environ.copy()
    env["ARXIV_QUERY"] = ARXIV_QUERY
    env["ARXIV_MAX_RESULTS"] = str(ARXIV_MAX_RESULTS)

    for script in SCRIPTS:
        run(script, env=env)

    print("\nfini!")

if __name__ == "__main__":
    main()
