
# git+https://github.com/HydroXai/ReNeLLM.git <- add to requirements

import time
import json
import os
import argparse
import random
from collections import namedtuple

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local application/library specific imports
from renellm.data_utils import data_reader
from renellm.prompt_rewrite_utils import shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange
from renellm.scenario_nest_utils import SCENARIOS
from renellm.harmful_classification_utils import harmful_classification


import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress warnings from a specific library
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


Result = namedtuple('Result', 'prompt targetModel error', defaults=(None, None, None))


def chat_with_llama(tokenizer, model, prompt, device):

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    output = model.generate(input_ids, max_new_tokens=64, do_sample=False, temperature=0)
    response = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)

    return response.strip()


def run_renellm(
    target_model: str,
    goal_prompt: str,
    iter_max: int,
    rewrite_model: str = "gpt-4o-mini-2024-07-18",
    judge_model: str = "gpt-4o-mini-2024-07-18",
    max_tokens: int = 3584,
    temperature: float = 0,
    round_sleep: int = 3,
    fail_sleep: int = 3,
    retry_times: int = 1000,
):

    args = argparse.Namespace(
            model_path=target_model,
            rewrite_model=rewrite_model,
            judge_model=judge_model,
            iter_max=iter_max,
            max_tokens=max_tokens,
            temperature=temperature,
            round_sleep=round_sleep,
            fail_sleep=fail_sleep,
            retry_times=retry_times,
            gpt_api_key=os.getenv("OPENAI_API_KEY"),
            gpt_base_url=None,
        )

    operations = [shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange]
    scenarios = SCENARIOS

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.use_default_system_prompt = False

    temp_harm_behavior = harm_behavior = goal_prompt

    loop_count = 0
    
    while True:
        print(
        "\n################################\n"
        f"Current Data: {goal_prompt}\n"
        f"Current Iteration Round: {loop_count+1}/{args.iter_max}\n"
        "################################\n")
        # prompt rewriting
        while True:
            print(f"******* Start Prompt Rewriting! *******")
            # randomly select the number of rewriting operations.
            n = random.randint(1, 6)
            # arrange the execution order randomly
            operation_indexes = random.sample(range(6), n)
            print(f"The number of rewriting functions is: {n}")
            print(f"The order of the rewriting operations is: {operation_indexes}\n")

            temp_rewrite_results = [['original prompt', temp_harm_behavior]]
            for index in operation_indexes:
                print(f"Excute function {index}: {operations[index].__name__}")
                harm_behavior = operations[index](args, harm_behavior)
                print(f"Rewritten prompt: {harm_behavior}\n")
                temp_rewrite_results.append([operations[index].__name__, harm_behavior])
                
            # check if the prompt meets the requirements; if it does, break out of the loop.
            print("******* Check if the prompt has been modified to be harmless *******")
            label = harmful_classification(args, harm_behavior)
            print(f"\nThe label given by {args.judge_model} is: {label}\n")
            if label == "1":
                print(f"******* Prompt Rewriting Done! *******\n")
                break
            else:
                print(f"******* Prompt Rewriting Failed! *******\n")
                harm_behavior = temp_harm_behavior
                continue

        # scenario nesting
        print(f"******* Start Prompt Scenario Nesting! *******")
        print(f"harm_behavior:\n{harm_behavior}\n")
        # randomly select a scenario
        scenario = random.choice(scenarios)
        nested_prompt = scenario.replace("<>", harm_behavior)
        print(f"nested_prompt:\n{nested_prompt}\n")

        print(f"******* Start Prompt Jailbreaking! *******")
        
        attack_output = chat_with_llama(tokenizer, model, nested_prompt, device)

        print(f"The output of the target model is:\n\n{attack_output}\n")

        loop_count += 1

        if "apologize" not in attack_output and "sorry" not in attack_output.lower():

            print(f"******* Determine if the attacked model's response to the nested prompt is harmful *******")
            jailbreak_label = harmful_classification(args, attack_output)
        
        else:
            if loop_count < args.iter_max:
                print(f"\nJailbreaking Prompt Failed!\n")
                harm_behavior = temp_harm_behavior
                continue
            else:
                print(f"\n******* Exceeded the maximum number of iterations {args.iter_max}, adopt the current round results and end the loop.*******\n")
                break

        if jailbreak_label == "1":
            print(f"\n******* Jailbreaking Prompt Successful! *******\n")
            break
        else:
            if loop_count < args.iter_max:
                print(f"\nJailbreaking Prompt Failed!\n")
                harm_behavior = temp_harm_behavior
                continue
            else:
                print(f"\n******* Exceeded the maximum number of iterations {args.iter_max}, adopt the current round results and end the loop.*******\n")
                break

    return nested_prompt, attack_output


def run(target_model: str, goal_prompt: str, result_path: str, iter_max: int = 20):
    try:
        adv_prompt, adv_response = run_renellm(target_model, goal_prompt, iter_max)
        target_model_name = os.path.basename(target_model)
        result = [Result(prompt=adv_prompt,
                        targetModel=target_model_name
                        )._asdict()]
    except Exception as e:
        result = [Result(prompt=goal_prompt,
                        error=f"An error was detected during the ReNeLLM attack: {e}")._asdict()]
    with open(result_path, 'w', encoding="utf8") as f:
        json.dump(result, f)


if __name__ == '__main__':
    fire.Fire(run)
    