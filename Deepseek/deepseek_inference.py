import json
import requests

# Read prompt and question from JSON
with open("deepseek_inference.json", "r") as file:
    data = json.load(file)

question = data["question"]
prompt = data["prompt"]


# DeepSeek API endpoint
url = "https://api.deepseek.com/v1/chat/completions"

# Your DeepSeek API key
api_key = "" # Replace with your actual API key

# Request headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Request body
payload = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ],
    "temperature": 0.7,
    "max_tokens": 512
}

# Send request to DeepSeek API
response = requests.post(url, headers=headers, json=payload)

# Handle response
if response.status_code == 200:
    result = response.json()
    reply = result["choices"][0]["message"]["content"]
    print("\nAssistant:", reply)
else:
    print("Error:", response.status_code)
    print(response.text)
