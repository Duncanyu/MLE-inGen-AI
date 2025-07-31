import requests
import json

API_URL = "http://127.0.0.1:8000/v1/completions"

prompt = "What is the capital of France?"

payload = {
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": prompt,
    "max_tokens": 50,
    "temperature": 0.7
}

response = requests.post(API_URL, headers={"Content-Type": "application/json"}, data=json.dumps(payload))

if response.status_code == 200:
    result = response.json()
    print(result["choices"][0]["text"].strip())
else:
    print(f"{response.status_code}: {response.text}")
