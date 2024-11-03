import json
from utils import load_jsonl, load_pickle, dump_pickle
import ast
from execution import evaluate_with_test_cases, evaluate_with_test_cases_serial
from tqdm import tqdm
import os
import sys
import logging
from copy import deepcopy
from openai import OpenAI
import re
import pickle
import javalang
import traceback
import numpy as np
import time
import multiprocessing
from concurrent.futures import as_completed, ProcessPoolExecutor
import tiktoken
data_type = sys.argv[1]
model_type = sys.argv[2]



code_top_threshold = 100
test_top_threshold = 50

runtime_dir = 'runtime'

times = f"GT_gpt{model_type}"

if model_type == '35':
    model_type = 'gpt-3.5-turbo-0613'
elif model_type == '4o':
    model_type = 'gpt-4o-2024-08-06'

encoding = tiktoken.encoding_for_model(model_type)




OPENAI_API_KEY = "your api key"


temperature = 0.7
top_p = 0.95
# gpt_tokens_limit = 100000
gpt_tokens_limits = {'gpt-4o': 125000, 'gpt-3.5':3000}

if 'gpt-4o' in model_type:
    gpt_tokens_limit = gpt_tokens_limits['gpt-4o']
elif 'gpt-3.5' in model_type:
    gpt_tokens_limit = gpt_tokens_limits['gpt-3.5']

if not os.path.exists(f'../{runtime_dir}/{data_type}/total_results/{times}'):
    already_files = []
else:
    already_files = os.listdir(f'../{runtime_dir}/{data_type}/total_results/{times}')
already_ids = [each.split('.')[0].replace('_', '/') for each in already_files]
print(already_ids)


def extract_python_tags(text, entry_point):
    pattern = r'```python(.*?)```'
    result = re.findall(pattern, text, flags=re.DOTALL)
    if len(result) < 1:
        return None
    
    final_results = []
    for cur_res in result:
        tmp = re.findall(r'def ' + re.escape(entry_point) + r'\([^)]*\)[^:\n]*:(.*)', cur_res, re.DOTALL)
        if len(tmp) == 1:
            lines = cur_res.splitlines()
            filtered_lines = [line for line in lines if not line.strip().startswith("assert")]
            final_results.append("\n".join(filtered_lines))
    if len(final_results) == 0:
        return None
    # wrong formats of code block
    # return tmp[0]
    return final_results[0]
def _repair(task_id, cur_code_id, cur_code, cur_test_id, consider_corrected_test, corrected_tests, messages, ff):
    current_tokens = 0
    for each in messages:
        try:
            current_tokens += len(encoding.encode(each['content']))
        except:
            current_tokens += len(each['content'].split())

    ff.write(f'>>>>>>>>>>>>>>>>>>>>>>fix test id: {cur_test_id}<<<<<<<<<<<<<<<<<<<<<<\n')
    entry_point = dataset_dict[task_id]['entry_point']
    multi_args = dataset_dict[task_id]['multi_args']
    if multi_args:
        consider_test_str = f"```python\nassert {entry_point}{consider_corrected_test['input']} == {consider_corrected_test['output']}\n```\n"
        corrected_tests_str = []
        for each in corrected_tests:
            cur_test = f"assert {entry_point}{each['input']} == {each['output']}"
            try:
                cur_test_tokens = len(encoding.encode(cur_test))
            except:
                cur_test_tokens = len(cur_test.split())
            if current_tokens + cur_test_tokens > gpt_tokens_limit:
                break
            current_tokens += cur_test_tokens
            corrected_tests_str.append(cur_test)
        corrected_tests_str = "\n".join(corrected_tests_str)
        corrected_tests_str = f"```python\n{corrected_tests_str}\n```\n"
    else:
        consider_test_str = f"```python\nassert {entry_point}({consider_corrected_test['input']}) == {consider_corrected_test['output']}\n```\n"
        corrected_tests_str = []
        for each in corrected_tests:
            cur_test = f"assert {entry_point}({each['input']}) == {each['output']}"
            try:
                cur_test_tokens = len(encoding.encode(cur_test))
            except:
                cur_test_tokens = len(cur_test.split())
            if current_tokens + cur_test_tokens > gpt_tokens_limit:
                break
            current_tokens += cur_test_tokens
            corrected_tests_str.append(cur_test)
        corrected_tests_str = "\n".join(corrected_tests_str)
        corrected_tests_str = f"```python\n{corrected_tests_str}\n```\n"

    instruction_fix = f"The generated code is not correct on the following test case, please fix the code to pass it and return the complete fixed code.\n{consider_test_str} Besides the above test case, the code should also pass the following test cases, which are already passed previously:\n{corrected_tests_str}.\n You only need to generate the fixed code, do not provide any assert statements or explanations."
    
    messages.append({"role": "user", "content": instruction_fix})
    ff.write("----------------------User Message----------------------\n")
    ff.write(f'{messages[-1]["content"]}\n')

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        completion = client.chat.completions.create(
            model=model_type,
            temperature=temperature,
            top_p=top_p,
            messages=messages
        )

        gpt_output = completion.choices[0].message.content
    except:
        logger.info(f'first openai api error: task_id:{task_id}, code_id:{cur_code_id} exception:{traceback.format_exc()}')
        time.sleep(2)
        try:
            completion = client.chat.completions.create(
                model=model_type,
                # response_format={ "type": "json_object" },
                temperature=temperature,
                top_p=top_p,
                messages=messages
            )
            gpt_output = completion.choices[0].message.content
        except:
            logger.info(f'second openai api error: task_id:{task_id}, code_id:{cur_code_id} exception:{traceback.format_exc()}')
            gpt_output = 'openai api error'

    ff.write("----------------------Assistant Message----------------------\n")
    ff.write(f'{gpt_output}\n')
    ff.flush()
    messages.append({"role": "assistant", "content": gpt_output})
    new_code = extract_python_tags(gpt_output, entry_point)
    return new_code, messages
def repair(cur_code, task_id, cur_code_id, cur_test_id, already_test_oracle_gt, messages, ff):

    global solution_dict
    global filter_code_ids
    global unknown_tests
    global code_test_res
    global code_test_output
    global code_fail_reason
    global logger
    consider_corrected_test = test_dict[cur_test_id]
    assert test_dict[cur_test_id]["output"] == already_test_oracle_gt[cur_test_id]
    corrected_tests = []
    for each in already_test_oracle_gt:
        if each == cur_test_id:
            continue
        # corrected_tests.append((test_dict[each]["input"], already_test_oracle_gt[each]))
        corrected_tests.append(test_dict[each])
        assert test_dict[each]["output"] == already_test_oracle_gt[each]
    
    new_code, new_messages = _repair(task_id, cur_code_id, cur_code, cur_test_id, consider_corrected_test, corrected_tests, messages, ff)
    # all_corrected_test_in_out = corrected_test_in_out + [consider_corrected_test_in_out]
    if new_code is None:
        # filter_code_ids.append(cur_code_id)
        logger.info(f'task_id:{task_id}, code_id:{cur_code_id} cannot return new code')
        return False, new_messages, False, False, False, False
    all_corrected_test_dict = {task_id: {}}
    for test_id in already_test_oracle_gt:
        all_corrected_test_dict[task_id][test_id] = test_dict[test_id]
        assert all_corrected_test_dict[task_id][test_id]['output'] == already_test_oracle_gt[test_id]

    solutions = []
    solutions.append(
        {
            "task_id": task_id,
            "prompt": dataset_dict[task_id]["prompt"],
            "entry_point": dataset_dict[task_id]["entry_point"],
            "solution_id": cur_code_id,
            "solution": new_code,
            "multi_args": dataset_dict[task_id]["multi_args"]
        }
    )

    corrected_code_test_res_raw = evaluate_with_test_cases_serial(solutions, all_corrected_test_dict, 1, assert_format=False, logger=logger, have_entry=True)
    corrected_code_test_res_raw = corrected_code_test_res_raw[0]
    corrected_code_test_output = corrected_code_test_res_raw['code_test_output']
    corrected_code_test_res = corrected_code_test_res_raw['code_test_res']
    corrected_code_test_fail_reason = corrected_code_test_res_raw['fail_reason']

    fix_flag = all(corrected_code_test_res.values())
    logger.info(f"<<<<<<<<<<<<<<<fail test begin<<<<<<<<<<<<<<<<<<<<<<<<<<")
    for each_test_id in corrected_code_test_res:
        if isinstance(corrected_code_test_fail_reason, str):
            cur_fail_reason = corrected_code_test_fail_reason
        else:
            cur_fail_reason = corrected_code_test_fail_reason[each_test_id]
        if corrected_code_test_res[each_test_id] == False:
            logger.info(f"fail test: task_id: {task_id}, code_id: {cur_code_id}, test_id: {each_test_id}, test_case: {all_corrected_test_dict[task_id][each_test_id]}, test_output: {corrected_code_test_output[each_test_id]}, fail_reason: {cur_fail_reason}")
    logger.info(f">>>>>>>>>>>>>>>fail test end>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    consider_test_res = corrected_code_test_res[cur_test_id]


    if not fix_flag:
        logger.info(f'task_id:{task_id}, code_id:{cur_code_id} fix failed, consider test: {consider_test_res}')
        return fix_flag, new_messages, False, False, False, False
    else:
        solution_dict[cur_code_id] = new_code
        unknown_test_dict = {task_id: {}}
        for each_un in unknown_tests:
            unknown_test_dict[task_id][each_un] = test_dict[each_un]

        un_code_test_res_raw = evaluate_with_test_cases_serial(solutions, unknown_test_dict, 1, assert_format=False, logger=logger, have_entry=True)  
        un_code_test_res_raw = un_code_test_res_raw[0]
        un_code_test_output = un_code_test_res_raw['code_test_output']
        un_code_test_res = un_code_test_res_raw['code_test_res']
        un_code_fail_reason = un_code_test_res_raw['fail_reason']

        for test_id in corrected_code_test_res:
            code_test_res[cur_code_id][test_id] = corrected_code_test_res[test_id]
            code_test_output[cur_code_id][test_id] = corrected_code_test_output[test_id]
            
            
            if isinstance(corrected_code_test_fail_reason, str):
                assign = corrected_code_test_fail_reason
            else:
                assign = corrected_code_test_fail_reason[test_id]


            code_fail_reason[cur_code_id][test_id] = assign

        other_tests_flag = True
        for test_id in un_code_test_res:
            code_test_res[cur_code_id][test_id] = un_code_test_res[test_id]
            code_test_output[cur_code_id][test_id] = un_code_test_output[test_id]
            
            if isinstance(un_code_fail_reason, str):
                assign = un_code_fail_reason
            else:
                assign = un_code_fail_reason[test_id]
            

            code_fail_reason[cur_code_id][test_id] = assign
            

            if un_code_test_res[test_id] == False:
                other_tests_flag = False
            
        logger.info(f'task_id:{task_id}, code_id:{cur_code_id} fix success, consider test: {consider_test_res}, other unknown tests: {other_tests_flag}')
        return fix_flag, new_messages, solution_dict[cur_code_id], code_test_res[cur_code_id], code_test_output[cur_code_id], code_fail_reason[cur_code_id]


def compute_code_tests(solution_dict, test_dict):
    for prompt in dataset:
        task_id = prompt["task_id"] 
        

        ast_tree = ast.parse(prompt["prompt"])
        for body in ast_tree.body:
            if isinstance(body, ast.FunctionDef) and body.name == prompt["entry_point"]:
                multi_args = len(body.args.args) > 1
                break

        for solution_id in solution_dict[task_id]:
            solutions.append(
                {
                    "task_id": task_id,
                    "prompt": prompt["prompt"],
                    "entry_point": prompt["entry_point"],
                    "solution_id": solution_id,
                    "solution": solution_dict[task_id][solution_id],
                    "multi_args": multi_args,
                }
            )

    code_test_res = evaluate_with_test_cases(solutions, test_dict, 0.01, assert_format=False, logger=logger)
    # Noted that task without spec code will not be included in the scores list
    dump_pickle(code_test_res, f"../{runtime_dir}/{data_type}/code_test_res.pkl")


def fix_parallel(each_code_id, cur_fix_path, cur_task_messages):
    global solution_dict
    cur_code = solution_dict[each_code_id]
    cur_fix_code_path = cur_fix_path + f'{each_code_id}'
    cur_task_messages = []
    if not cur_task_messages:
        instruction_gen = "I want you to act like a Python programmer. I will give you the declaration of a function and comments about its property. You need to implement the body of the function in the code block. Do not modify any code I provide.\n"
        messages = [{"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": instruction_gen},
                    {"role": "assistant", "content": "OK, I'm ready to help."},
            ]

        question_gen = "Here is the question.\n" + "```python\n" + dataset_dict[task_id]['prompt'] + "```\n"
        
        messages.append({"role": "user", "content": question_gen})

        
        initial_code = "```python\n" + dataset_dict[task_id]['prompt'] + solution_dict[each_code_id] + "\n```\n"
        messages.append({"role": "assistant", "content": initial_code})



        # cur_task_messages[each_code_id] = messages
        if not os.path.exists(cur_fix_code_path):
            ff = open(cur_fix_code_path, 'w')
            ff.write(f'task_id:{task_id}\n')
            ff.write('----------------------System Message----------------------\n')
            ff.write(f'{messages[0]["content"]}\n')
            ff.write('----------------------User Message----------------------\n')
            ff.write(f'{messages[1]["content"]}\n')
            ff.write('----------------------Assistant Message----------------------\n')
            ff.write(f'{messages[2]["content"]}\n')
            ff.write('----------------------User Message----------------------\n')
            ff.write(f'{messages[3]["content"]}\n')
            ff.write('----------------------Assistant Message----------------------\n')
            ff.write(f'{messages[4]["content"]}\n')
            ff.flush()
        else:
            ff = open(cur_fix_code_path, 'a')
    else:
        ff = open(cur_fix_code_path, 'a')
        messages = cur_task_messages

    fix_flag, new_messages, cur_solution_dict, cur_code_test_res, cur_code_test_output, cur_code_fail_reson = repair(cur_code, task_id, each_code_id, consider_test_id, already_test_oracle_gt, messages, ff)

    return each_code_id, fix_flag, new_messages, cur_solution_dict, cur_code_test_res, cur_code_test_output, cur_code_fail_reson



if __name__ == '__main__':
    require_num = 1 
    
    dataset = load_jsonl(f"../data/{data_type}.jsonl")
    # [{"task_id": , "prompt": , "entry_point": , "canonical_solution":, "test":}]
    solutions = load_jsonl(f"../{runtime_dir}/{data_type}/solutions.jsonl")
    # [{"task_id": , "prompt": , "completions":[solution1, solution2, ..., solution200]}]
    tests = load_pickle(f"../runtime/{data_type}/test_cases.pkl")
    # [{"task_id": , "prompt": , "tc_input_output":[{input: , output: , tc_str: }, ...]}]

    print("start construct dataset_dict")
    dataset_dict = {}

    logger1 = logging.getLogger()
    while len(logger1.handlers) > 0:
        logger1.handlers.pop()
    logging.basicConfig(
        format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    for i in tqdm(range(len(dataset))):
        prompt = dataset[i]
        task_id = prompt["task_id"] 
        multi_args = None
        # determine whether is multi_args

        ast_tree = ast.parse(prompt["prompt"])
        for body in ast_tree.body:
            if isinstance(body, ast.FunctionDef) and body.name.strip() == prompt["entry_point"].strip():
                multi_args = len(body.args.args) > 1
                break
        if multi_args == None:
            multi_args = False
            logger1.info(f'task_id:{task_id} cannot determine whether is multi_args')
        dataset[i]["multi_args"] = multi_args
        dataset_dict[task_id] = dataset[i]
    total_test_dict = {}
    # dict{task_id: {test_id: {input: , output: , tc_str: }}}
    print("start construct total_test_dict")
    for each in tqdm(tests):
        task_id = each["task_id"]
        total_test_dict[task_id] = {}
        cur_tests = each["tc_input_output"][:test_top_threshold]
        # cur_tests = each["tc_input_output"][:100]
        for i in range(len(cur_tests)):
            test_id = f't-{i}'
            total_test_dict[task_id][test_id] = cur_tests[i]

    total_solution_dict = {}
    # dict{task_id: {solution_id: solution}}
    for each in tqdm(solutions):
        task_id = each["task_id"]
        total_solution_dict[task_id] = {}
        # cur_solutions = each["completions"][:200]
        cur_solutions = each["completions"][:code_top_threshold]
        for i in range(len(cur_solutions)):
            solution_id = f's-{i}'
            total_solution_dict[task_id][solution_id] = cur_solutions[i]

    canonical_solutions = []
    for prompt in dataset:
        task_id = prompt["task_id"] 
        canonical_solutions.append(
            {
                "task_id": task_id,
                "prompt": prompt["prompt"],
                "entry_point": prompt["entry_point"],
                "solution_id": -1,
                "solution": prompt["canonical_solution"],
                "multi_args": prompt["multi_args"]
            }
        )

    print('start execute canonical code on test')

    if not os.path.exists(f"../runtime/{data_type}/canonical_code_test_res_raw.pkl"):
        logger1 = logging.getLogger()
        while len(logger1.handlers) > 0:
            logger1.handlers.pop()
        logging.basicConfig(
            format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        canonical_code_test_res_raw = evaluate_with_test_cases(canonical_solutions, total_test_dict, 1, assert_format=False, logger=logger1)
        # dict{task_id: {test_id: output}}
        dump_pickle(canonical_code_test_res_raw, f"../runtime/{data_type}/canonical_code_test_res_raw.pkl")
    canonical_code_test_res_raw = load_pickle(f"../runtime/{data_type}/canonical_code_test_res_raw.pkl")
    canonical_code_test_output = {}
    for each in canonical_code_test_res_raw:
        task_id = each["task_id"]
        canonical_code_test_output[task_id] = each['code_test_output']
    canonical_code_test_res = {}
    for each in canonical_code_test_res_raw:
        task_id = each["task_id"]
        canonical_code_test_res[task_id] = each["code_test_res"]
    canonical_fail_reason = {}
    for each in canonical_code_test_res_raw:
        task_id = each["task_id"]
        canonical_fail_reason[task_id] = each["fail_reason"]

    total_human_test_oracle_gt_dict = {}
    # dict{task_id: {test_id: oracle}}
    for task_id in total_test_dict:
        total_human_test_oracle_gt_dict[task_id] = {}
        for test_id in total_test_dict[task_id]:
            total_human_test_oracle_gt_dict[task_id][test_id] = canonical_code_test_output[task_id][test_id]


    solutions = []
    for prompt in dataset:
        task_id = prompt["task_id"] 
        for solution_id in total_solution_dict[task_id]:
            solutions.append(
                {
                    "task_id": task_id,
                    "prompt": prompt["prompt"],
                    "entry_point": prompt["entry_point"],
                    "solution_id": solution_id,
                    "solution": total_solution_dict[task_id][solution_id],
                    "multi_args": prompt["multi_args"]
                }
            )
    if not os.path.exists(f"../{runtime_dir}/{data_type}/code_test_res.pkl"):
        logger1 = logging.getLogger()
        while len(logger1.handlers) > 0:
            logger1.handlers.pop()
        logging.basicConfig(
            format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        total_code_test_res_raw = evaluate_with_test_cases(solutions, total_test_dict, 1, assert_format=False, logger=logger1)
        dump_pickle(total_code_test_res_raw, f"../{runtime_dir}/{data_type}/code_test_res.pkl")
    total_code_test_res_raw = load_pickle(f"../{runtime_dir}/{data_type}/code_test_res.pkl")



    total_code_test_output_all = {}
    # dict{task_id: {solution_id: {test_id: output}}}
    total_code_test_res_all = {}
    # dict{task_id: {solution_id: {test_id: res}}}
    total_fail_reason_all = {}
    for each in total_code_test_res_raw:
        task_id = each["task_id"]
        solution_id = each["solution_id"]
        code_test_output = each["code_test_output"]
        code_test_res = each["code_test_res"]
        fail_reason = each["fail_reason"]
        code_test_output = dict(list(code_test_output.items())[:test_top_threshold])
        code_test_res = dict(list(code_test_res.items())[:test_top_threshold])
        if task_id not in total_code_test_output_all:
            total_code_test_output_all[task_id] = {}
            total_code_test_res_all[task_id] = {}
            total_fail_reason_all[task_id] = {}
        total_code_test_output_all[task_id][solution_id] = code_test_output
        total_code_test_res_all[task_id][solution_id] = code_test_res
        total_fail_reason_all[task_id][solution_id] = fail_reason

    total_code_test_output = {}
    total_code_test_res = {}
    total_fail_reason = {}
    
    for task_id in total_code_test_res_all:
        total_code_test_output[task_id] = {}
        total_code_test_res[task_id] = {}
        total_fail_reason[task_id] = {}
        cur_code_test_res_all = total_code_test_res_all[task_id]
        passed_tests_num = {}
        for solution_id in cur_code_test_res_all:
            # print(cur_code_test_res_all[solution_id].values())
            passed_tests_num[solution_id] = sum(cur_code_test_res_all[solution_id].values())
        
        passed_tests_num = dict(sorted(passed_tests_num.items(), key=lambda x:int(x[0].split('-')[1])))
        for solution_id in list(passed_tests_num.keys())[:code_top_threshold]:
            total_code_test_output[task_id][solution_id] = total_code_test_output_all[task_id][solution_id]
            total_code_test_res[task_id][solution_id] = total_code_test_res_all[task_id][solution_id]
            total_fail_reason[task_id][solution_id] = total_fail_reason_all[task_id][solution_id]


    for task_id in total_fail_reason:
        for solution_id in total_fail_reason[task_id]:
            if isinstance(total_fail_reason[task_id][solution_id], str):
                total_fail_reason[task_id][solution_id] = {}
                for test_id in total_test_dict[task_id]:
                    total_fail_reason[task_id][solution_id][test_id] = total_fail_reason[task_id][solution_id]


    total_result_path = f'../{runtime_dir}/{data_type}/total_results/{times}/'
    if not os.path.exists(total_result_path):
        os.makedirs(total_result_path)
    for task_id in tqdm(total_solution_dict):
        print(f'start task_id:{task_id}')    
        if os.path.exists(f"../{runtime_dir}/{data_type}/fix_process_gpt/{times}/{task_id.replace('/', '_')}"):
            os.system(f"rm -r ../{runtime_dir}/{data_type}/fix_process_gpt/{times}/{task_id.replace('/', '_')}")
        try:

            logger_path = f'../{runtime_dir}/{data_type}/logger/{times}/'
            if not os.path.exists(logger_path):
                os.makedirs(logger_path)
            logger = logging.getLogger()
            while len(logger.handlers) > 0:
                logger.handlers.pop()
            logger_file = logger_path + f'{task_id.replace("/", "_")}.log'
            logging.basicConfig(
                format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                level=logging.INFO,
                filename=logger_file,
                filemode='w'  
            )
            logging.getLogger("openai._base_client").disabled = True
            logging.getLogger("httpx").disabled = True

            

            cur_fix_path = f'../{runtime_dir}/{data_type}/fix_process_gpt/{times}/{task_id.replace("/", "_")}/'
            # if os.path.exists(cur_fix_path):
            #     os.system(f'rm -r {cur_fix_path}')
            if not os.path.exists(cur_fix_path):
                os.makedirs(cur_fix_path)



            solution_dict = total_solution_dict[task_id]
            test_dict = total_test_dict[task_id]
            human_test_oracle_gt_dict = total_human_test_oracle_gt_dict[task_id]
            code_test_output = total_code_test_output[task_id]
            # {solution_id: {test_id: output}}
            code_test_res = total_code_test_res[task_id]
            code_fail_reason = total_fail_reason[task_id]
            # {solution_id: {test_id: res}}
            delete_tests = []
            unknown_tests = [] 
            for test_id in test_dict:
                unknown_tests.append(test_id)
            
            filter_code_ids = [] 
            already_test_oracle_gt = {} 
        
            iter_num = 0
            cur_total_result = {}
            cur_task_messages = {}
            while len(unknown_tests) > 0:
                if all(x in filter_code_ids for x in list(code_test_output.keys())):
                    break
                iter_num += 1
                logger.info(f"task_id:{task_id}:{iter_num} iteration")

                # update rank
                pass_num = {}
                for each_un in unknown_tests:
                    pass_num[each_un] = 0
                    for each_code_id in code_test_output:
                        if each_code_id in filter_code_ids:
                            continue
                        if isinstance(code_test_res[each_code_id][each_un], bool):
                            if code_test_res[each_code_id][each_un]:
                                pass_num[each_un] += 1
                        else:
                            if code_test_res[each_code_id][each_un].all():
                                pass_num[each_un] += 1
                rank = dict(sorted(pass_num.items(), key=lambda x:x[1], reverse=False))

                consider_test_id = list(rank.keys())[0]


                for test_id in rank:
                    if test_id == consider_test_id:
                        continue
                    assert rank[test_id] >= rank[consider_test_id]
                
                
                
                cur_fail_reason = canonical_fail_reason[task_id][consider_test_id]
                if cur_fail_reason is not None:
                    logger.info(f'---ref_code_fail --test_id:{consider_test_id} ---task_id:{task_id}, epoch: {iter_num}')
                    delete_tests.append(consider_test_id)
                    unknown_tests.remove(consider_test_id)
                    test_dict.pop(consider_test_id)
                    human_test_oracle_gt_dict.pop(consider_test_id)
                    already_satisfied_ids = [] 
                    for cur_code_id in code_test_output:
                        code_test_output[cur_code_id].pop(consider_test_id)
                        code_test_res[cur_code_id].pop(consider_test_id)
                        code_fail_reason[cur_code_id].pop(consider_test_id)
                        if cur_code_id in filter_code_ids:
                            continue
                        if all(code_test_res[cur_code_id].values()):
                            already_satisfied_ids.append(cur_code_id)


                    if len(already_satisfied_ids) >= require_num:
                        find_enough_codes = True
                        pass_num_dict = {}

                        
                        for cur_code_id in code_test_output:
                            if cur_code_id in filter_code_ids:
                                continue
                            pass_num_dict[cur_code_id] = []
                            for test_id in code_test_output[cur_code_id]:
                                if code_test_res[cur_code_id][test_id]:
                                    pass_num_dict[cur_code_id].append(test_id)
                        save_pass_num_dict = deepcopy(pass_num_dict)
                        save_pass_num_dict = dict(sorted(save_pass_num_dict.items(), key=lambda x:len(x[1]), reverse=True))
                        sorted_pass_num_dict = dict(sorted(pass_num_dict.items(), key=lambda x:len(x[1]), reverse=True)[:10])
                        for each in sorted_pass_num_dict:
                            sorted_pass_num_dict[each] = len(sorted_pass_num_dict[each])
                        total_test_num = len(test_dict)
                        logger.info(f'---enough ids from bad precondition, find_enough_codes: {find_enough_codes} ---task_id:{task_id}, epoch: {iter_num}, pass_num_dict:{sorted_pass_num_dict}, total_test_num:{total_test_num}')

                        break
                    continue

                cur_input = test_dict[consider_test_id]["input"]
                cur_oracle = test_dict[consider_test_id]["output"]
                cur_oracle_gt = human_test_oracle_gt_dict[consider_test_id]
                test_dict[consider_test_id]["output"] = cur_oracle_gt
                unknown_tests.remove(consider_test_id)    
                already_test_oracle_gt[consider_test_id] = cur_oracle_gt

                pass_num_dict = {}
                cur_fix_ids = [] 

                outputs_on_test = []
                
                fail_reasons = []
                any_need_fix = False
                all_same = False
                for cur_code_id in code_test_output:
                    if cur_code_id in filter_code_ids:
                        continue
                    cur_code_test_output = code_test_output[cur_code_id][consider_test_id]
                    if isinstance(code_fail_reason[cur_code_id], str):
                        fail_reasons.append(code_fail_reason[cur_code_id])
                    else:
                        fail_reasons.append(code_fail_reason[cur_code_id][consider_test_id])
                    
                    if isinstance(cur_code_test_output, np.ndarray):
                        if any((cur_code_test_output == np.array(t)).all() for t in outputs_on_test):
                            pass
                        else:
                            outputs_on_test.append(cur_code_test_output)
                    else:
                        if cur_code_test_output not in outputs_on_test:
                            outputs_on_test.append(cur_code_test_output)
                    
                    if isinstance(cur_code_test_output, np.ndarray):
                        if (cur_code_test_output == np.array(cur_oracle_gt)).all():
                            pass
                        else:
                            any_need_fix = True
                    else:
                        if cur_code_test_output != cur_oracle_gt:
                            any_need_fix = True
                if len(outputs_on_test) == 1 and all([x == None for x in fail_reasons]):
                    iter_num -= 1
                    all_same = True
                    test_dict[consider_test_id]["output"] = outputs_on_test[0]
                    already_test_oracle_gt[consider_test_id] = outputs_on_test[0]
                    already_satisfied_ids = []
                    for cur_code_id in code_test_output:
                        if cur_code_id in filter_code_ids:
                            continue
                        code_test_res[cur_code_id][consider_test_id] = True
                        if all(code_test_res[cur_code_id].values()):
                            already_satisfied_ids.append(cur_code_id)

                        pass_num_dict[cur_code_id] = []
                        for test_id in code_test_output[cur_code_id]:
                            if code_test_res[cur_code_id][test_id]:
                                pass_num_dict[cur_code_id].append(test_id)
                    save_pass_num_dict = deepcopy(pass_num_dict)
                    save_pass_num_dict = dict(sorted(save_pass_num_dict.items(), key=lambda x:len(x[1]), reverse=True))
                    
                    sorted_pass_num_dict = dict(sorted(pass_num_dict.items(), key=lambda x:len(x[1]), reverse=True)[:10])
                    for each in sorted_pass_num_dict:
                        sorted_pass_num_dict[each] = len(sorted_pass_num_dict[each])
                    total_test_num = len(test_dict)
                    find_enough_codes = False
                    if len(already_satisfied_ids) >= require_num:
                        find_enough_codes = True
                    
                    logger.info(f'---skip iteration, any_need_fix:{any_need_fix} after check, need fix {len(cur_fix_ids)} codes, find_enough_codes: {find_enough_codes}---task_id:{task_id}, epoch: {iter_num}, pass_num_dict:{sorted_pass_num_dict}, total_test_num:{total_test_num}')
                    if find_enough_codes:
                        break
                    continue
                already_satisfied_ids = []
                for cur_code_id in code_test_output:
                    if cur_code_id in filter_code_ids:
                        continue
                    cur_code_test_output = code_test_output[cur_code_id][consider_test_id]
                    # print(cur_code_test_output, cur_oracle_gt, cur_oracle)
                    if cur_code_test_output == cur_oracle_gt:
                        code_test_res[cur_code_id][consider_test_id] = True
                        if all(code_test_res[cur_code_id].values()):
                            already_satisfied_ids.append(cur_code_id)
                    else:
                        code_test_res[cur_code_id][consider_test_id] = False
                        cur_fix_ids.append(cur_code_id)
                    pass_num_dict[cur_code_id] = []
                    for test_id in code_test_output[cur_code_id]:
                        if code_test_res[cur_code_id][test_id]:
                            pass_num_dict[cur_code_id].append(test_id)
                


                save_pass_num_dict = deepcopy(pass_num_dict)
                save_pass_num_dict = dict(sorted(save_pass_num_dict.items(), key=lambda x:len(x[1]), reverse=True))
                
                sorted_pass_num_dict = dict(sorted(pass_num_dict.items(), key=lambda x:len(x[1]), reverse=True)[:10])
                for each in sorted_pass_num_dict:
                    sorted_pass_num_dict[each] = len(sorted_pass_num_dict[each])
                total_test_num = len(test_dict)
                find_enough_codes = False
                if len(already_satisfied_ids) >= require_num:
                    find_enough_codes = True

                
                logger.info(f'---after check, need fix {len(cur_fix_ids)} codes, find_enough_codes: {find_enough_codes}---task_id:{task_id}, epoch: {iter_num}, pass_num_dict:{sorted_pass_num_dict}, total_test_num:{total_test_num}')
                if find_enough_codes:
                    break
                if len(cur_fix_ids) == 0:
                    continue

                cur_fix_ids = sorted(cur_fix_ids, key=lambda x:len(pass_num_dict[x]), reverse=True)


                with ProcessPoolExecutor(30) as executor:
                    futures = []
                    for each_code_id in tqdm(cur_fix_ids):
                        if each_code_id in filter_code_ids:
                            continue
                        if each_code_id in cur_task_messages:
                            messages = cur_task_messages[each_code_id]
                        else:
                            messages = []
                        args = (each_code_id, cur_fix_path, messages)
                        future = executor.submit(fix_parallel, *args)
                        futures.append(future)
                    already_satisfied_ids = []
                    for cur_code_id in code_test_output:
                        if cur_code_id in cur_fix_ids: 
                            continue
                        if all(code_test_res[cur_code_id].values()):
                            already_satisfied_ids.append(cur_code_id)
                            if len(already_satisfied_ids) >= require_num:
                                find_enough_codes = True
                                break
                    
                    new_filter_code_ids = []
                    for idx, future in enumerate(as_completed(futures)):
                        each_code_id, fix_flag, new_messages, cur_solution_dict, cur_code_test_res, cur_code_test_output, cur_code_fail_reason = future.result()
                        # cur_task_messages[each_code_id] = new_messages
                        if not fix_flag:
                            # print('finish one', each_code_id)   
                            new_filter_code_ids.append(each_code_id) 
                            continue
                        solution_dict[each_code_id] = cur_solution_dict
                        code_test_res[each_code_id] = cur_code_test_res
                        code_test_output[each_code_id] = cur_code_test_output
                        code_fail_reason[each_code_id] = cur_code_fail_reason
                        if all(code_test_res[each_code_id].values()):
                            already_satisfied_ids.append(each_code_id)
                            if len(already_satisfied_ids) >= require_num:
                                find_enough_codes = True
                                break
                        # print('finish one', each_code_id)                        
                    not_filtered_num = len([x for x in list(code_test_output.keys()) if x not in filter_code_ids])
                    if len(new_filter_code_ids) == len(cur_fix_ids) == not_filtered_num: # or len(cur_fix_ids)
                        logger.info(f'---no fix correct, discard test --test_id:{consider_test_id} ---task_id:{task_id}, epoch: {iter_num}')
                        delete_tests.append(consider_test_id)
                        test_dict.pop(consider_test_id)
                        already_test_oracle_gt.pop(consider_test_id)
                        # human_test_oracle_gt_dict.pop(consider_test_id)
                        for cur_code_id in code_test_output:
                            code_test_output[cur_code_id].pop(consider_test_id)
                            code_test_res[cur_code_id].pop(consider_test_id)
                            if cur_code_id in filter_code_ids:
                                continue
                            if all(code_test_res[cur_code_id].values()):
                                already_satisfied_ids.append(cur_code_id)
                    else:
                        filter_code_ids.extend(new_filter_code_ids)
                        





                pass_num_dict = {}
                
                for cur_code_id in code_test_output:
                    if cur_code_id in filter_code_ids:
                        continue
                    pass_num_dict[cur_code_id] = []
                    for test_id in code_test_output[cur_code_id]:
                        if code_test_res[cur_code_id][test_id]:
                            pass_num_dict[cur_code_id].append(test_id)
                save_pass_num_dict = deepcopy(pass_num_dict)
                save_pass_num_dict = dict(sorted(save_pass_num_dict.items(), key=lambda x:len(x[1]), reverse=True))
                sorted_pass_num_dict = dict(sorted(pass_num_dict.items(), key=lambda x:len(x[1]), reverse=True)[:10])
                for each in sorted_pass_num_dict:
                    sorted_pass_num_dict[each] = len(sorted_pass_num_dict[each])
                total_test_num = len(test_dict)
                logger.info(f'---after fix, find_enough_codes: {find_enough_codes}---task_id:{task_id}, epoch: {iter_num}, pass_num_dict:{sorted_pass_num_dict}, total_test_num:{total_test_num}')
                if find_enough_codes:
                    break
            cur_total_result["task_id"] = task_id
            cur_total_result['iter_num'] = iter_num
            cur_total_result["already_test_oracle_gt"] = already_test_oracle_gt
            cur_total_result["pass_num_dict"] = save_pass_num_dict
            cur_total_result["code_test_output"] = code_test_output
            cur_total_result["code_test_res"] = code_test_res
            cur_total_result["solution_dict"] = solution_dict
            try:
                json.dump(cur_total_result, open(f'{total_result_path}/{task_id.replace("/", "_")}.json', 'w'), indent=4)
            except:
                open(f'{total_result_path}/{task_id.replace("/", "_")}.txt', 'w').write(str(cur_total_result))
                pickle.dump(cur_total_result, open(f'{total_result_path}/{task_id.replace("/", "_")}.pkl', 'wb'))



        except Exception as e:
            logger.error(f'task_id:{task_id} error: {traceback.format_exc()}')

