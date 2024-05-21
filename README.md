# gpt-llama-index-speed-test
Speed test between GPT-4, GPT-4o, and GPT-4o using LlamaIndex. 

One interesting result of this testing is that the order that models that are run (in theory) in parallel actually matters. This is surprising for me at least. Specifically the ordering of the model in the initial list and the model chosen for the if/else statement logic seems to affect performance, at least in Google Colab where these tests were originally run (I haven't checked using PyCharm or other client IDE): 

    models_to_run_in_parallel = ["gpt-4o_llx","gpt-4o"]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for model in models_to_run_in_parallel:
            if model == "gpt-4o_llx":
                futures[executor.submit(query_llama_openai, user_input)] = model
            else:
                futures[executor.submit(query_openai, model, user_input)] = model

So a [randomized test] (https://github.com/wjleece/gpt-llama-index-speed-test/blob/main/randomized-speed-test.py) was also created to counteract these effects.

