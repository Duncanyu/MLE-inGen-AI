from pathlib import Path
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_DIR   = Path(__file__).resolve().parent
REPORTS_DIR= BASE_DIR / "reports"
PROMPTS_IN = BASE_DIR / "prompts.json"
OUT_PATH   = REPORTS_DIR / "generations.json"

BASELINE = "HuggingFaceH4/zephyr-7b-beta"
LORA_DIR = BASE_DIR / "outputs" / "zephyr7b_lora"
FULL_DIR = None
MAX_NEW  = 256
TEMP     = 0.7
TOP_P    = 0.9

def load_prompts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["prompts"] if isinstance(data, dict) else data

def gen(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=TOP_P)
    return tok.decode(out[0], skip_special_tokens=True)

def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    prompts = load_prompts(PROMPTS_IN)

    tok0 = AutoTokenizer.from_pretrained(BASELINE, use_fast=True); tok0.pad_token = tok0.eos_token
    m0   = AutoModelForCausalLM.from_pretrained(BASELINE, device_map="auto")

    tok1 = AutoTokenizer.from_pretrained(str(LORA_DIR), use_fast=True); tok1.pad_token = tok1.eos_token
    base1= AutoModelForCausalLM.from_pretrained(BASELINE, device_map="auto")
    m1   = PeftModel.from_pretrained(base1, str(LORA_DIR))

    m2 = tok2 = None
    if FULL_DIR:
        tok2 = AutoTokenizer.from_pretrained(str(FULL_DIR), use_fast=True); tok2.pad_token = tok2.eos_token
        m2   = AutoModelForCausalLM.from_pretrained(str(FULL_DIR), device_map="auto")

    rows=[]
    for p in prompts:
        rows.append({
            "prompt": p,
            "baseline": gen(m0, tok0, p),
            "lora":     gen(m1, tok1, p),
            "full":     gen(m2, tok2, p) if m2 is not None else None
        })

    OUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
