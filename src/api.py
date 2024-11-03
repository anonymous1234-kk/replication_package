from openai import OpenAI
import os
import json
from tqdm import tqdm
import time
import re
import traceback
import tiktoken
def num_tokens_from_messages(encoding, messages, model):
    """Return the number of tokens used by a list of messages."""

    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
def infer_llm(data_type, engine, instruction, exemplars, query, answer_num=5, max_tokens=2048, diversity_gen_num=1):
    """
    Args:
        instruction: str
        exemplars: list of dict {"query": str, "answer": str}
        query: str
    Returns:
        answers: list of str
    """
    OPENAI_API_KEY = 'your_api_key'
    client = OpenAI(api_key=OPENAI_API_KEY)


    messages = [{"role": "system", "content": "You are a helpful AI assistant.."},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "OK, I'm ready to help."},
        ]
    
    
    messages.append({"role": "user", "content": query})
    if engine == "gpt-3.5-turbo-0613":
        try:
            encoding = tiktoken.encoding_for_model(engine)
        except KeyError:
            print("Warning: model not found. Using o200k_base encoding.")
            encoding = tiktoken.get_encoding("o200k_base")
        remain_tokens = 4096 - num_tokens_from_messages(encoding, messages, engine)
        # print(remain_tokens)
        max_tokens = remain_tokens
    else:
        max_tokens = max_tokens * diversity_gen_num + 1000
    while True:
        try:
            answers = client.chat.completions.create(
                model=engine,
                messages=messages,
                temperature=0.8,
                max_tokens=max_tokens,
                top_p=0.95,
                n=answer_num,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return [response.message.content for response in answers.choices if response.finish_reason != 'length']
        except:
            print(f'retry, {traceback.format_exc()}')
            time.sleep(2)


