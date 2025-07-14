from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI()


system_prompt= """
You are an ai assistnat who is ecpert in breaking down the complex problem and then resolve 
For the given user input, analyse the input and break the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.
The steps are you get a user input you alalyze you agin think for several think and them return an output with explanation and then finally you validate the output as will before giving you final output
Follow the steps in sequence that is "analtse" , "think", "output","validate" and finally "result".
Rules:
1. Follow the strinct JSON output as per output schema
2. always perform one at a time and wait for input
3. Carefully analyse the user query
Output format:
{{steo:"string" , content:"String"}}
Example:
Input: What is 2+2
Output: {{step:"analyse" , content:"Alright! the user is interested in maths query and he is asking a basic arithmetic operation}}
Output :{{step :"think", content:"to perform the addition i must go from left to right and all operand"}}
Output :{{step:"output" , content:"4"}}
Output :{{step:"validate" , content:"seems like 4 is correct answer for 2+2"}}
Output :{{step:"result" , content:"2+2=4 and that is calulated by adding all number"}}
"""

result= client.chat.completions.create(
    model="gpt-4o",
    response_format={"type":"json_object"},
    messages=[
        {"role":"system", "content":system_prompt},
        {"role":"user", "content":"what is 3+4*5"},
        {"role":"assistant", "step": "analyse", "content": "The user has asked a mathematical question involving both addition and multiplication. According to the order of operations, multiplication should be performed before addition." },
        {"role":"assistant" , "step":"analyse", "content":"The user is asking for the result of the expression 3 + 4 * 5, which involves both addition and multiplication."}
    ]
)
print(result.choices[0].message.content)