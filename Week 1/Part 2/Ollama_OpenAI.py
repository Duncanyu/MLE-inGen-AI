from openai import OpenAI

client = OpenAI(base_url = "http://localhost:11434/v1", api_key = "ollama")

response = client.chat.completions.create(
  model="llama3",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "When was the first iPhone released?"},
    {"role": "assistant", "content": "The first iPhone was released in 2007."},
    {"role": "user", "content": "When was the first Android running phone released?"},
    {"role": "assistant", "content": "The first Android phone was released in 2008"},
    {"role": "user", "content": "How long was the time difference between the release of the two phones?"}
  ]
)

print(response.choices[0].message.content)