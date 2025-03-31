from openai import OpenAI
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


def chatCompletion(llm, messages, temperature, retry_times, round_sleep, fail_sleep, api_key, base_url=None):
    input_ids = llm.tokenizer.apply_chat_template(messages,
                                                add_generation_prompt=True,
                                                return_tensors="pt").to(llm.model.device)
    generated_ids = llm.model.generate(input_ids,
                                    max_new_tokens=512,
                                    do_sample=True,
                                    pad_token_id=llm.tokenizer.pad_token_id,
                                    temperature=temperature)
    outputs = generated_ids[0][input_ids.shape[-1]:]
    response = llm.tokenizer.decode(outputs, skip_special_tokens=True)
    return response


# for gpt
# def chatCompletion(model, messages, temperature, retry_times, round_sleep, fail_sleep, api_key, base_url=None):
#     if base_url is None:
#         client = OpenAI(
#             api_key=api_key
#             )
#     else:
#         client = OpenAI(
#         api_key=api_key,
#         base_url=base_url
#     )
#     try:
#         response = client.chat.completions.create(
#                 model=model,
#                 messages=messages,
#                 temperature=temperature
#             )
#     except Exception as e:
#         print(e)
#         for retry_time in range(retry_times):
#             retry_time = retry_time + 1
#             print(f"{model} Retry {retry_time}")
#             time.sleep(fail_sleep)
#             try:
#                 response = client.chat.completions.create(
#                     model=model,
#                     messages=messages,
#                     temperature=temperature
#                 )
#                 break
#             except:
#                 continue

#     model_output = response.choices[0].message.content.strip()
#     time.sleep(round_sleep)

#     return model_output

# for claude
def claudeCompletion(model, max_tokens, temperature, prompt, retry_times, round_sleep, fail_sleep, api_key, base_url=None):
    if base_url is None:
        client = Anthropic(
            api_key=api_key
            )
    else:
        client = Anthropic(
            base_url=base_url,
            auth_token=api_key
            )   
    try:
        completion = client.completions.create(
            model=model,
            max_tokens_to_sample=max_tokens,
            temperature=temperature,
            prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
            )
    except Exception as e:
        print(e)
        for retry_time in range(retry_times):
            retry_time = retry_time + 1
            print(f"{model} Retry {retry_time}")
            time.sleep(fail_sleep)
            try:
                completion = client.completions.create(
                model=model,
                max_tokens_to_sample=max_tokens,
                temperature=temperature,
                prompt=prompt
                )
                break
            except:
                continue

    model_output = completion.completion.strip()
    time.sleep(round_sleep)

    return model_output
    