from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()

response = client.chat.completions.create(
    model = "gpt-4o",
    messages=[
        {"role" : "system", "content" : "you are AI assistant that tell the whether of different places with help of context , context : the tempurature of lucknow is 33 degree celcius"},
        {"role":"user", "content":"what is the temperature of lucknow "}
    ]
)

print(response.choices[0].message.content)