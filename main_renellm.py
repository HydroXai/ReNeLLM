
# git+https://github.com/HydroXai/ReNeLLM.git <- add to requirements

import argparse
import fire
import json
import logging
import os
import random
import torch
import traceback
import warnings

from accelerate import Accelerator
from collections import namedtuple
from renellm.prompt_rewrite_utils import shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange
from renellm.scenario_nest_utils import SCENARIOS
from renellm.harmful_classification_utils import harmful_classification
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerFast


warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


MAX_RETRY_COUNT = 2
MAX_ITERATION = 20


accelerator = Accelerator()
quantization_config = BitsAndBytesConfig(
    load_in_8Bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)


Result = namedtuple('Result', 'response error', defaults=(None, None))


def run_renellm(
    target_model,
    rewrite_model,
    goal_prompt: str,
    iter_max: int,
    max_tokens: int = 3584,
    temperature: float = 0.8,
    round_sleep: int = 3,
    fail_sleep: int = 3,
    retry_times: int = 1000,
):

    args = argparse.Namespace(
            # model_path=target_model,
            target_model=target_model,
            rewrite_model=rewrite_model,
            judge_model=rewrite_model,
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

    temp_harm_behavior = harm_behavior = goal_prompt

    loop_count = 0

    while True:
        logging.debug(
        "\n################################\n"
        f"Current Data: {goal_prompt}\n"
        f"Current Iteration Round: {loop_count+1}/{iter_max}\n"
        "################################\n")
        # prompt rewriting
        while True:
            logging.debug(f"******* Start Prompt Rewriting! *******")
            # randomly select the number of rewriting operations.
            n = random.randint(1, 6)
            # arrange the execution order randomly
            operation_indexes = random.sample(range(6), n)
            logging.debug(f"The number of rewriting functions is: {n}")
            logging.debug(f"The order of the rewriting operations is: {operation_indexes}\n")

            temp_rewrite_results = [['original prompt', temp_harm_behavior]]
            for index in operation_indexes:
                logging.debug(f"Excute function {index}: {operations[index].__name__}")
                harm_behavior = operations[index](args, harm_behavior)
                logging.debug(f"Rewritten prompt: {harm_behavior}\n")
                temp_rewrite_results.append([operations[index].__name__, harm_behavior])

            # check if the prompt meets the requirements; if it does, break out of the loop.
            logging.debug("******* Check if the prompt has been modified to be harmless *******")
            label = harmful_classification(args, harm_behavior)
            logging.debug(f"\nThe label given by {args.judge_model} is: {label}\n")
            if label == "1":
                logging.debug(f"******* Prompt Rewriting Done! *******\n")
                break
            else:
                logging.debug(f"******* Prompt Rewriting Failed! *******\n")
                harm_behavior = temp_harm_behavior
                continue

        # scenario nesting
        logging.debug(f"******* Start Prompt Scenario Nesting! *******")
        logging.debug(f"harm_behavior:\n{harm_behavior}\n")
        # randomly select a scenario
        scenario = random.choice(scenarios)
        nested_prompt = scenario.replace("<>", harm_behavior)
        logging.debug(f"nested_prompt:\n{nested_prompt}\n")

        logging.debug(f"******* Start Prompt Jailbreaking! *******")
        attack_output = target_model.generate(nested_prompt)

        logging.debug(f"The output of the target model is:\n\n{attack_output}\n")

        loop_count += 1
        attack_output_lower = attack_output.lower()

        if "apologize" not in attack_output_lower and "sorry" not in attack_output_lower:
            logging.debug(f"******* Determine if the attacked model's response to the nested prompt is harmful *******")
            jailbreak_label = harmful_classification(args, attack_output)
            if jailbreak_label == "1":
                logging.debug(f"\n******* Jailbreaking Prompt Successful! *******\n")
                break

        if loop_count >= iter_max:
            logging.debug(f"\n******* Exceeded the maximum number of iterations {iter_max}, adopt the current round results and end the loop.*******\n")
            break

        logging.debug(f"\nJailbreaking Prompt Failed!\n")
        harm_behavior = temp_harm_behavior

    return nested_prompt, attack_output


class LLM(object):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto")
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt) -> str:
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(input_ids,
                                        max_new_tokens=512,
                                        do_sample=True,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        temperature=0.8)
        outputs = generated_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return response


def main(input_path: str, output_path: str) -> None:
    results = []
    with open(input_path, mode='r') as f:
        samples = json.load(f)

        target_model = LLM(samples.get("targetModelPath"))
        rewrite_model = LLM(samples.get("rewriteModelPath"))

        for goal_prompt in samples.get("goalPrompts"):
            count = 0
            while count < MAX_RETRY_COUNT:
                try:
                    adv_prompt, adv_response = run_renellm(target_model, rewrite_model, goal_prompt, MAX_ITERATION)
                    results.append(Result(response=adv_prompt)._asdict())
                    break
                except Exception as e:
                    print(traceback.format_exc())

                    if count == MAX_RETRY_COUNT - 1:
                        results = [Result(error=f"An error was detected during the ReNeLLM attack: {e}")._asdict()]
                    count += 1
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(results, f)


if __name__ == '__main__':
    fire.Fire(main)
