# from openai import OpenAI
# # remove: import openai

from langfuse.openai import openai
from langfuse import observe
from dotenv import load_dotenv
import json
import os

import requests

load_dotenv()

client = openai.Client()
@observe
def get_weather(city : str):

    # TODO!: Do an actual API call
    print("this function is calling now : get_weather" , city)

    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The wether in {city} is {response.text}."
    return "Something wrong"
@observe
def run_command(command):
   res = os.system(command=command)
   return res

available_tools = {
    "get_weather " : {
        "fn" : get_weather,
        "description" : "Takes a city name as an input and returns the current weather for the city"
    },

    "run_command" : {
        "fn" : run_command,
        "description" : "Take a command as an inout and execute it on the system and return output"
    }
}


system_prompt = f"""
    You are an helpfull AI Assistant who is specialized in resolving user query.
    You work on start, plan, action, observe mode.
    For the given user query and available tools, plan the step by step execution, based on the planning,
    select the relevant tool from the available tool. and based on the tool selection you perform an action to call the tool.
    Wait for the observation and based on the observation from the tool call resolve the user query.

    Rules:
    - Follow the Output JSON Format.
    - Always perform one step at a time and wait for next input
    - Carefully analyse the user query

    Output JSON Format:
    {{
        "step": "string",
        "content": "string",
        "function": "The name of function if the step is action",
        "input": "The input parameter for the function",
    }}

    Available Tools:
    - get_weather: Takes a city name as an input and returns the current weather for the city
    - run_command: Takes a command as input to execute on system and returns ouput

     Example:
    User Query: What is the weather of new york?
    Output: {{ "step": "plan", "content": "The user is interseted in weather data of new york" }}
    Output: {{ "step": "plan", "content": "From the available tools I should call get_weather" }}
    Output: {{ "step": "action", "function": "get_weather", "input": "new york" }}
    Output: {{ "step": "observe", "output": "12 Degree Cel" }}
    Output: {{ "step": "output", "content": "The weather for new york seems to be 12 degrees." }}


"""

messages = [
    {"role" : "system" , "content" : system_prompt}
]

query = input("> ")
messages.append({"role" : "user" , "content" : query})

while True:
    response= client.chat.completions.create(
    model="gpt-4o",
    response_format={"type":"json_object"},
    messages=messages
    )
    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role" : "assistant" , "content" : json.dumps(parsed_response)})
    if parsed_response.get("step") == "plan":
        print(f"🧠 : {parsed_response.get("content")}")
        continue
    if parsed_response.get("step") == "action":
        tool_name = parsed_response.get("function")
        tool_input = parsed_response.get("input")

        if available_tools.get(tool_name,False) != False:
            output = available_tools[tool_name].get("fn")(tool_input)
            messages.append({"role" : "assistant" , "content" : json.dumps({"step" : "observe" , "output" : output})})    
            continue

        
    if parsed_response.get("step") != "output":
        print(f"🧠 : {parsed_response.get("content")}")
        continue
    print(f"🤖 : {parsed_response.get("content")}")
    break
