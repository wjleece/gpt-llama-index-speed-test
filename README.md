# gpt-llama-index-speed-test
Speed test between GPT-4, GPT-4o, and GPT-4o using LlamaIndex. 

One interesting result of this testing is that the order that models that are run (in theory) in parallel actually matters. This is surprising for me at least. Specifically the ordering of the model in the initial list and the model chosen for the if/else statement logic seems to affect performance, at least in Google Colab where these tests were originally run. 

In other words 

    models_to_run_in_parallel = ["gpt-4o_llx","gpt-4o"]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for model in models_to_run_in_parallel:
            if model == "gpt-4o_llx":
                futures[executor.submit(query_llama_openai, user_input)] = model
            else:
                futures[executor.submit(query_openai, model, user_input)] = model

gives generally shows gpt-4o to complete more quickly, which is what you'd expect given that its not taking the extra time to call LlamaIndex libraries.

But surprisingly

    models_to_run_in_parallel = ["gpt-4o", "gpt-4o_llx"]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        for model in models_to_run_in_parallel:
            if model == "gpt-4o:
                futures[executor.submit(query_llama_openai, user_input)] = model
            else:
                futures[executor.submit(query_openai, model, user_input)] = model

generially shows that GPT-4o with LlamaIndex ("gpt-4o_llx") completes its tasks more quickly, which is certainly counter intuitive. 

Thus it seems that order matters.

So a [randomized test](https://github.com/wjleece/gpt-llama-index-speed-test/blob/main/randomized-speed-test.py) was also created to counteract these effects to enable a clearer determination of how much LlamaIndex slows down operations. 

The results of the randomized tests show that GPT-4o alone runs slightly faster than GPT-4o with LlamaIndex, which is what we'd expect.

The results also show that GPT-4o is generally more than 2x as fast as GPT-4, which matches OpenAI's performance benchmarks for the model. 

