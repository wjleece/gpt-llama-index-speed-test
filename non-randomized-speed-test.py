import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI as LlamaOpenAI
import nest_asyncio
nest_asyncio.apply()



# Get org ID & API key from files
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the script
api_key_file = os.path.join(script_dir, "api-key.txt")  # Construct the path to the api-key.txt file

# Read the API key from the file
with open(api_key_file, "r") as file:
    api_key = file.read().strip()  # strip() removes any leading/trailing whitespace

# Set the environment variable
os.environ["OPENAI_API_KEY"] = api_key


temperature = 0  # Set to 0 to minimize variability of responses

temperature = 0  # Set to 0 to minimize variability of responses

def query_openai(model, user_input):
    client = OpenAI()
    start_time = time.time()
    stream = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ],
        stream=True,
    )

    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content

    end_time = time.time()
    elapsed_time = end_time - start_time
    return model, response, elapsed_time

def query_llama_openai(user_input):
    client = LlamaOpenAI(temperature=temperature, model="gpt-4o")
    start_time = time.time()
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant"),
        ChatMessage(role="user", content=user_input, stream=True),
    ]
    resp = client.stream_chat(messages)
    response = ""
    for chunk in resp:
        response += chunk.delta
    end_time = time.time()
    elapsed_time = end_time - start_time
    return "gpt-4o_llx", response, elapsed_time

user_input = input("How can I help you today?\n")

#Run GPT-4 test first
model="gpt-4"
model, response, g4_time = query_openai(model, user_input)
print(f"\n\n{model} took {g4_time:.2f} seconds to answer this question.\n")
print(response)

# Run GPT-4o tests (with and without LlamaIndex) in parallel
models_to_run_in_parallel = ["gpt-4o", "gpt-4o_llx"]

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {}
    for model in models_to_run_in_parallel:
        if model == "gpt-4o":
            futures[executor.submit(query_openai, model, user_input)] = model
        else:
            futures[executor.submit(query_llama_openai, user_input)] = model

    results = {}
    for future in as_completed(futures):
        model = futures[future]
        try:
            model, response, elapsed_time = future.result()
            print(f"\n\n{model} took {elapsed_time:.2f} seconds to answer this question.\n")
            print(response)
            results[model] = elapsed_time
        except Exception as exc:
            print(f"{model} generated an exception: {exc}")

# Compare performance
g4o_time = results["gpt-4o"]
g4o_llx_time = results["gpt-4o_llx"]

print(f"\nGPT-4o took {g4o_time:.2f} seconds to answer this question.")
print(f"GPT-4o using LlamaIndex took {g4o_llx_time:.2f} seconds to answer this question.")
print(f"\nGPT-4o completed the task in {((g4o_time / g4_time)) * 100:.2f}% of the time that GPT-4 took to complete it, or {1 / (g4o_time / g4_time):.2f} times faster.")
print(f"GPT-4o using LlamaIndex completed the task in {((g4o_llx_time / g4_time)) * 100:.2f}% of the time that GPT-4 took to complete it, or {1 / (g4o_llx_time / g4_time):.2f} times faster.")
