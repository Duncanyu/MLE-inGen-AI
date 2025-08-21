import os, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

BASE_MODEL  = "meta-llama/Meta-Llama-3-8B"
CHATML_PATH = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/data/chatml.jsonl"
OUT_DIR     = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week5/Homework/outputs/llama3_full"
CUTOFF_LEN  = 2048
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
EPOCHS      = 1
LR          = 1e-5
WARMUP      = 0.03
WEIGHT_DECAY= 0.0
LOG_STEPS   = 10
SAVE_STEPS  = 200
SEED        = 42
DEEPSPEED   = None

def load_chatml_jsonl(path):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            convo=json.loads(line)
            user=assistant=""
            for m in convo:
                if m["role"]=="user": user=m["content"]
                elif m["role"]=="assistant": assistant=m["content"]
            rows.append({"prompt":user,"response":assistant})
    return Dataset.from_list(rows)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ds  = load_chatml_jsonl(CHATML_PATH)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        bf16=True,
        deepspeed=DEEPSPEED,
        optim="adamw_torch",
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tok,
        packing=True,
        max_seq_length=CUTOFF_LEN,
        formatting_func=lambda ex: [f"<|user|>\n{ex['prompt']}\n<|assistant|>\n{ex['response']}"],
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    print(f"saved full tto {OUT_DIR}")

if __name__ == "__main__":
    main()