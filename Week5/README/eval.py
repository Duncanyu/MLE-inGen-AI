from pathlib import Path
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

def load_prompts(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["prompts"] if isinstance(data, dict) else data

def gen(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=TOP_P)
    return tok.decode(out[0], skip_special_tokens=True)

def load_baseline(model_id):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto")
    return model, tok

def load_lora(base_id, adapter_dir):
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype)
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=quant, device_map="auto")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    return model, tok

def load_full(full_dir):
    tok = AutoTokenizer.from_pretrained(str(full_dir), use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(full_dir), device_map="auto")
    return model, tok

def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    prompts = load_prompts(PROMPTS_IN)

    m0, t0 = load_baseline(BASELINE)
    m1, t1 = load_lora(BASELINE, LORA_DIR)
    m2 = t2 = None
    if FULL_DIR:
        m2, t2 = load_full(FULL_DIR)

    rows = []
    for p in prompts:
        rows.append({
            "prompt": p,
            "baseline": gen(m0, t0, p),
            "lora": gen(m1, t1, p),
            "full": gen(m2, t2, p) if m2 is not None else None
        })

    OUT_PATH.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote to {OUT_PATH}")

if __name__ == "__main__":
    main()
