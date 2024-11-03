import json
import os
import re
from tqdm import tqdm
import time
import ast
import pickle
from utils import load_jsonl, load_pickle, dump_jsonl, dump_pickle
import shutil
from concurrent.futures import as_completed, ProcessPoolExecutor
import sys
from api import infer_llm, infer_llm_completion
from execution import evaluate_spec_with_test_cases, evaluate_with_specs_and_casual_input, evaluate_with_test_cases
from _execution import time_limit
import time


one_type_num = 150
diversity_gen_num = 5


def produce_parallel(id):
    global dataset
    global data_type

    prompt = dataset[id]['prompt']
    task_id = dataset[id]['task_id']
    entry_point = dataset[id]['entry_point']

    task_all_completions = []
    start_time = time.time()
    # generate prompt
        
    diversity_prompt = f"Please implement the function {diversity_gen_num} times, introducing diversity in the generated solutions."
    instruction, question = get_prompt(id, diversity_prompt)
    # inference with chatgpt api
    completions = []
    times = 0
    while True:
        times += 1
        if times > 10:
            print('retry', times)
        answers = infer_llm(data_type, llm_engine, instruction, None, question, answer_num=30, max_tokens=2048, diversity_gen_num=diversity_gen_num)
        for answer in answers:
            pattern = r'```python(.*?)```'
            result = re.findall(pattern, answer, flags=re.DOTALL)
            if len(result) < 1:
                continue
            
            for cur_res in result:
                tmp = re.findall(r'^def ' + re.escape(entry_point.strip()) + r'\([^)]*\)[^:\n]*:(.*)', cur_res, re.DOTALL | re.MULTILINE)
                if len(tmp) == 1:
                    lines = cur_res.splitlines()
                    filtered_lines = [line for line in lines if not line.strip().startswith("assert")]
                    completions.append("\n".join(filtered_lines))
            if len(completions) >= one_type_num:
                break
        if len(completions) >= one_type_num:
            break
        time.sleep(1)
    # print('task_id:', task_id, 'diversity_key:', diversity_key, 'completions:', len(completions))
    task_all_completions.extend(completions)
        # write to file
    end_time = time.time()
    print('task_id:', task_id, 'time:', end_time - start_time, 'completions:', len(task_all_completions))
    assert len(task_all_completions)  >= 100
    return task_id, prompt, task_all_completions


def get_prompt(id, diversity_prompt):
    """
    Prompt to generate exec code with no exemplars
    """
    global dataset
    instruction = f"I want you to act like a Python programmer. I will give you the declaration of a function and comments about its property. You need to implement the body of the function in the code block. {diversity_prompt} Each time, please implement the body of the function and include the complete import statements and function signature in separate code blocks. Each piece of code should be enclosed in a single ```python``` code block. Do not surround anything else with ```python``` code blocks except code. Do not modify any code I provide. Do not provide any explanations.\n"

    question = "Here is the question.\n" + "```python\n" + dataset[id]['prompt'] + "```\n"

    return instruction, question

def inference_all(gen_model='none', resume_id=None):
    # gen_model: 4o
    """
    Generate exec codes for all samples in dataset,
    """
    # load human_eval_dataset

    global dataset
    file_mode = 'a'
    resume_id = resume_id if resume_id is not None else 0


    final_output = {}


    f = open(f"../{runtime_dir}/{data_type}/solutions.jsonl", file_mode)
    with ProcessPoolExecutor(20) as executor:
        futures = []
        for id in tqdm(range(resume_id, len(dataset))):
            if dataset[id]['task_id'] in already_task_ids:
                continue
            print(dataset[id]['task_id'])
            args = (id,)
            future = executor.submit(produce_parallel, *args)
            futures.append(future)
        for idx, future in tqdm(enumerate(as_completed(futures))):
            task_id, prompt, task_all_completions = future.result()
            f.write(json.dumps({'task_id': task_id, 'prompt': prompt, 'completions': task_all_completions}) + "\n")
            f.flush()
            
    f.close()        

if __name__ == '__main__':
    data_type = sys.argv[1]
    gen_model = sys.argv[2]
    if '35' in gen_model:
        llm_engine = "gpt-3.5-turbo-0613"
    elif '4o' in gen_model:
        llm_engine = "gpt-4o-2024-08-06"
    total_start_time = time.time()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open(f"../data/{data_type}.jsonl", 'r') as f:
        dataset = [json.loads(line) for line in f.readlines()]


    if '35' in gen_model:
        runtime_dir = "runtime"
    else:
        runtime_dir = f"runtime_{gen_model}"
    if not os.path.exists(f"../{runtime_dir}/{data_type}"):
        os.makedirs(f"../{runtime_dir}/{data_type}")

    already_task_ids = []
    if os.path.exists(f"../{runtime_dir}/{data_type}/solutions.jsonl"):
        current_solutions = load_jsonl(f"../{runtime_dir}/{data_type}/solutions.jsonl")
        for each in current_solutions:
            already_task_ids.append(each['task_id'])


    inference_all(gen_model=gen_model, resume_id=None)
    total_end_time = time.time()
    print(total_end_time - total_start_time)
 
