import argparse, json, random
from datasets import load_dataset

DATASET_NAME = "tatsu-lab/alpaca"
DATASET_SPLIT = "train"
MAX_SAMPLES  = 2000
SEED         = 42
OUT_PATH     = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/data/chatml.jsonl"

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default=DATASET_NAME)
    ap.add_argument("--split", default=DATASET_SPLIT)
    ap.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()

    ds = load_dataset(args.dataset_name, split=args.split)
    rows = list(ds)
    random.Random(args.seed).shuffle(rows)
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows[:args.max_samples]

    out = args.out
    import os; os.makedirs("data", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in rows:
            f.write(json.dumps(to_chatml(s), ensure_ascii=False) + "\n")
    print(f"WRote {len(rows)} ChatML conversations to {out}")

if __name__ == "__main__":
    main()
