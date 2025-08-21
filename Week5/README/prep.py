from pathlib import Path
import os, argparse, json, random
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_PATH = DATA_DIR / "chatml.jsonl"

DATASET_NAME = "tatsu-lab/alpaca"
DATASET_SPLIT = "train"
MAX_SAMPLES   = 2000
SEED          = 42

def to_chatml(sample):
    instr = sample.get("instruction") or sample.get("question") or sample.get("input") or ""
    input_txt = sample.get("input") or ""
    output = sample.get("output") or sample.get("text") or sample.get("answer") or ""
    user_msg = instr if instr else input_txt
    if input_txt and instr and instr not in user_msg:
        user_msg = f"{instr}\n\n{input_txt}"
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": str(user_msg).strip()},
        {"role": "assistant", "content": str(output).strip()},
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    ds = load_dataset(args.dataset_name, split=args.split)
    rows = list(ds)
    random.Random(args.seed).shuffle(rows)
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows[:args.max_samples]

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")
    print(f"WWrote {len(rows)} chatml conversations to {out_path}")

if __name__ == "__main__":
    main()
