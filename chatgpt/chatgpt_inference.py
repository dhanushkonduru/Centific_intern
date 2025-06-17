import json
from openai import OpenAI

with open("chatgpt_inference.json", "r") as file:
    data = json.load(file)

question = data["question"]
print(question)
prompt=data["prompt"]
print(prompt)
client = OpenAI(api_key="") # Your OpenAI API key 
response = client.chat.completions.create(
     model="gpt-4", 
     messages=[ 
         {"role": "system", "content": prompt}, 
         {"role": "user", "content": question} ] ) 
print(response) 
summary = response.choices[0].message.content 
print(summary)