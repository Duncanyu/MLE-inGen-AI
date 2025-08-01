import time
import requests
import json
from openai import OpenAI
from keys import OPENAI_KEY

API_URL = "http://127.0.0.1:8000/v1/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
RESULTS_FILE = "task3_results.txt"

client = OpenAI(api_key=OPENAI_KEY)

tests = [
    ("What is the capital of France?", "Paris"),
    ("2 + 2 =", "4"),
    ("What is the capital of Japan?", "Tokyo"),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen"),
    ("What is the boiling point of water (in Celsius)?", "100"),
    ("What is 9 * 8?", "72"),
    ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the chemical symbol for gold?", "Au"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("What is the square root of 144?", "12")
]

def query_vllm(prompt):
    start = time.time()
    payload = {"model": MODEL_NAME, "prompt": prompt, "max_tokens": 50, "temperature": 0}
    response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
    elapsed = time.time() - start

    if response.status_code == 200:
        text = response.json()["choices"][0]["text"].strip()
        return text, elapsed
    else:
        return None, elapsed

def query_chatgpt(prompt):
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.7
    )
    output = response.choices[0].message.content
    elapsed_time = round(time.time() - start_time, 2)
    return output, elapsed_time

vllm_correct = 0
chatgpt_correct = 0

results = []
results.append("llm test results\n")

for i, (prompt, expected) in enumerate(tests, 1):
    print(f"Test {i}/{len(tests)}: {prompt}")
    results.append(f"Test {i}: {prompt}\n")

    vllm_output, vllm_time = query_vllm(prompt)
    is_correct_vllm = expected.lower() in vllm_output.lower() if vllm_output else False
    results.append(f"  vllm: {vllm_output} | time: {vllm_time:.2f}s | correct: {is_correct_vllm}\n")
    if is_correct_vllm:
        vllm_correct += 1

    # == da gpt
    chatgpt_output, chatgpt_time = query_chatgpt(prompt)
    is_correct_chatgpt = expected.lower() in chatgpt_output.lower()
    results.append(f"  chat: {chatgpt_output} | time: {chatgpt_time:.2f}s | correct: {is_correct_chatgpt}\n\n")
    if is_correct_chatgpt:
        chatgpt_correct += 1

summary = f"\nsummary:\n  vllm correct: {vllm_correct}/{len(tests)}\n  got correct: {chatgpt_correct}/{len(tests)}"
print(summary)
results.append(summary)

with open(RESULTS_FILE, "w") as f:
    f.writelines(results)

print(f"\saved to {RESULTS_FILE}")
