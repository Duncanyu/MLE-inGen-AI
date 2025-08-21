from pathlib import Path
import os, json, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
OUT_DIR     = BASE_DIR / "outputs" / "zephyr7b_lora"
CHATML_PATH = DATA_DIR / "chatml.jsonl"

BASE_MODEL   = "HuggingFaceH4/zephyr-7b-beta"
CUTOFF_LEN   = 2048
BATCH_SIZE   = 1
GRAD_ACCUM   = 8
EPOCHS       = 1
LR           = 2e-4
WARMUP       = 0.03
WEIGHT_DECAY = 0.0
LOG_STEPS    = 10
SAVE_STEPS   = 200
SEED         = 42

USE_4BIT     = True
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGET  = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def load_chatml_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            convo = json.loads(line)
            user, assistant = "", ""
            for m in convo:
                role = m.get("role")
                if role == "user":
                    user = m.get("content", "")
                elif role == "assistant":
                    assistant = m.get("content", "")
            rows.append({"prompt": user, "response": assistant})
    return Dataset.from_list(rows)

def format_example(ex):
    return f"<|user|>\n{ex['prompt']}\n<|assistant|>\n{ex['response']}"

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_chatml_jsonl(CHATML_PATH)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    ds_text = ds.map(lambda ex: {"text": format_example(ex)})
    def tok_fn(batch):
        out = tok(
            batch["text"],
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
        )
        out["labels"] = out["input_ids"].copy()
        return out
    ds_tok = ds_text.map(tok_fn, batched=True, remove_columns=ds_text.column_names)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=compute_dtype,
            device_map="auto",
        )

    model.config.use_cache = False
    if getattr(model.config, "pretraining_tp", None) is not None:
        model.config.pretraining_tp = 1

    lcfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lcfg)

    targs = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        seed=SEED,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds_tok,
        tokenizer=tok,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    print(f"saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
