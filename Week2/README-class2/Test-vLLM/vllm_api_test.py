import requests
import json

API_URL = "http://127.0.0.1:8000/v1/completions"

prompt = """Question: Which of the following is the capital of Canada?
A. Paris
B. New York
C. Toronto
D. Ottawa
Answer:"""

payload = {
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": prompt,
    "max_tokens": 10,
    "temperature": 0
}

response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

if response.status_code == 200:
    result = response.json()
    answer = result["choices"][0]["text"].strip()

    print(f"Answer: {answer}")

    with open("vllm_inference_proof.txt", "w") as f:
        f.write(f"Prompt:\n{prompt}\n\n")
        f.write(f"Answer: {answer}\n")

else:
    print(f"{response.status_code}: {response.text}")