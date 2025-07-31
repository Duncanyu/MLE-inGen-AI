from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
prompt = "Apple is a company that"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(result)
with open("task2_proof.txt", "w") as f:
    f.write(f"Task 2 run at: {datetime.now()}\n")
    f.write(f"Prompt: {prompt}\n")
    f.write(f"Output: {result}\n")