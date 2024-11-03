import os
from math import comb
from execution import evaluate_with_test_code
from utils import load_jsonl, load_pickle, dump_jsonl, dump_pickle
from _evaluation import _estimate_pass_at_k
import ast
from tqdm import tqdm
import json
import re
import logging
import multiprocessing
import time
import numpy as np
import sys
def merge_completion_and_prompt(total_solution_dict, dataset_dict, isplus=False):

    res = []
    for task_id in dataset_dict:
        cur_data = dataset_dict[task_id]
        for solution_id in total_solution_dict[task_id]:
            cur_code = total_solution_dict[task_id][solution_id]
            tmp = re.findall(r'def ' + re.escape(cur_data["entry_point"]) + r'\([^)]*\)[^:\n]*:(.*)', cur_code, re.DOTALL)
            if len(tmp) == 1:
                have_entry = True
            else:
                have_entry = False

            res.append(
                {
                    "task_id": task_id,
                    "solution_id":solution_id,
                    "prompt": cur_data["prompt"],
                    "completion": cur_code,
                    "test": cur_data["test"] if not isplus else cur_data["plus_test"],
                    "entry_point": cur_data["entry_point"],
                    "have_entry": have_entry
                }
            )
    return res

def extend_sorted(code_sorted, completions, rd=True):
    """
    extend the `data` to include all solutions, so that data length is 200
    Args:
        code_sorted: the sorted solutions with score. The deduplicated version, length is less equal than 200
        completions: The non-deduplicated version, length is always 200
    """
    scores = {}
    for data in completions:
        if data["task_id"] not in code_sorted:
            scores[data["task_id"]] = [(x, 0) for x in data["completions"]]
            continue
        scores[data["task_id"]] = []
        for x in data["completions"]:
            for dt in code_sorted[data["task_id"]]:
                if dt[0] == x:
                    scores[data["task_id"]].append((x, dt[1] if not rd else round(dt[1], 4)))
                    break
            else:
                scores[data["task_id"]].append((x, 0))
        scores[data["task_id"]] = sorted(scores[data["task_id"]], key=lambda x: x[1], reverse=True)
    return scores

def extend_ispassed(ispassed, completions):
    for data in completions:
        for x in data["completions"]:
            if x not in ispassed[data["task_id"]]:
                ispassed[data["task_id"]][x] = 0
    return ispassed

def original_pass_at_k(data_type, isplus=False):
    se_path = "solution_eval_plus.pkl" if isplus else "solution_eval.pkl"
    if os.path.exists(f"../runtime/{data_type}/{se_path}"):
        data = load_pickle(f"../runtime/{data_type}/{se_path}")
    else:
        completions = load_jsonl(f"../runtime/{data_type}/solutions.jsonl")
        prompts = load_jsonl(f"../data/{data_type}.jsonl" if not isplus else f"../data/{data_type}_plus.jsonl")

        data = evaluate_with_test_code(merge_completion_and_prompt(completions, prompts, isplus), 1)
        dump_pickle(data, f"../runtime/{data_type}/{se_path}")

    completions = load_jsonl(f"../runtime/{data_type}/solutions.jsonl")
    ispassed = {}
    for dt in data:
        if dt["task_id"] not in ispassed:
            ispassed[dt["task_id"]] = {}
        ispassed[dt["task_id"]][dt["completion"]] = dt["passed"]

    corrects = []
    totals = []
    for dt in completions:
        totals.append(len(dt["completions"]))
        corrects.append(sum([ispassed[dt["task_id"]][x] if x in ispassed[dt["task_id"]] else 0 for x in dt["completions"]]))
    
    pass_at_k = {}
    for k in [1,2,5]:
        pass_at_k[f"pass@{k}"] = round(_estimate_pass_at_k(totals, corrects, k).mean() * 100, 2)
    # print(pass_at_k)
    return pass_at_k
def task(task_id, cur_path, ):
    open('tmp/%s'%task_id.replace('/','_'), 'w')
    start_time = time.time()
    cur_results = {}
    cur_see_pass_num = {}
    cur_simple_pass_num_dict = {}
    cur_solution_dict = {}
    if cur_path.endswith('.pkl'):
        cur_result = load_pickle(cur_path)
    else:
        cur_result = json.load(open(cur_path))

    cur_results[task_id] = cur_result
    pass_num_dict = cur_result['pass_num_dict']
    for k in range(len(pass_num_dict) - 1):
        assert len(list(pass_num_dict.values())[k]) >= len(list(pass_num_dict.values())[k + 1])
    simple_pass_num = [] 
    see_simple_pass_num = []
    for solution_id in pass_num_dict:
        simple_pass_num.append((solution_id, len(pass_num_dict[solution_id])))
        see_simple_pass_num.append(len(pass_num_dict[solution_id]))
    cur_see_pass_num[task_id] = see_simple_pass_num
    cur_simple_pass_num_dict[task_id] = simple_pass_num
    solution_dict = cur_result['solution_dict']
    cur_solution_dict[task_id] = solution_dict
    end_time = time.time()
    os.system('rm tmp/%s'%task_id.replace('/','_'))
    return cur_results, cur_see_pass_num, cur_simple_pass_num_dict, cur_solution_dict

def sorted_pass_at_k_top_hat_n(data_type, isplus=False, times=""):
    if 'o1' not in times:
        if data_type == 'human_eval' and isplus:
            data_type = 'human_eval_plus'
    if times:
        se_path = f"fixed_solution_eval_{times}_plus.pkl" if isplus else f"fixed_solution_eval_{times}.pkl"
    else:
        se_path = "fixed_solution_eval_plus.pkl" if isplus else "fixed_solution_eval.pkl"
    if data_type == 'human_eval' and isplus:
        dataset = load_jsonl(f"../data/human_eval_plus.jsonl")
    else:
        dataset = load_jsonl(f"../data/{data_type}.jsonl")
    dataset_dict = {}
    for i in tqdm(range(len(dataset))):
        prompt = dataset[i]
        task_id = prompt["task_id"] 
        
        ast_tree = ast.parse(prompt["prompt"])
        for body in ast_tree.body:
            if isinstance(body, ast.FunctionDef) and body.name == prompt["entry_point"]:
                multi_args = len(body.args.args) > 1
                break
        dataset[i]["multi_args"] = multi_args
        dataset_dict[task_id] = dataset[i]
    if data_type == 'human_eval' and isplus:
        data_path = f"../runtime/human_eval_plus/{se_path}"
    else:
        data_path = f"../runtime/{data_type}/{se_path}"
    if os.path.exists(data_path):
        data = load_pickle(data_path)
        total_results = {}
        total_see_pass_num = {}
        total_simple_pass_num_dict = {}
        total_solution_dict = {}
        


        pool = multiprocessing.Pool(processes=200)
        processs = []
        
        for task_id in tqdm(dataset_dict):
            if times:
                if os.path.exists(f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.pkl"):
                    cur_path = f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.pkl"
                else:
                    cur_path = f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.json"
            else:
                if os.path.exists(f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.pkl"):
                    cur_path = f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.pkl"
                else:
                    cur_path = f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.json"
            p = pool.apply_async(task, args=(task_id, cur_path))
            processs.append(p)
        
        for each in processs:
            cur_results, cur_see_pass_num, cur_simple_pass_num_dict, cur_solution_dict = each.get()
            for task_id in cur_results:
                total_results[task_id] = cur_results[task_id]
            for task_id in cur_see_pass_num:
                total_see_pass_num[task_id] = cur_see_pass_num[task_id]
            for task_id in cur_simple_pass_num_dict:
                total_simple_pass_num_dict[task_id] = cur_simple_pass_num_dict[task_id]
            for task_id in cur_solution_dict:
                total_solution_dict[task_id] = cur_solution_dict[task_id]


    else:


        total_results = {}
        total_see_pass_num = {}
        total_simple_pass_num_dict = {}
        total_solution_dict = {}
        


        pool = multiprocessing.Pool(processes=200)
        processs = []
        
        for task_id in tqdm(dataset_dict):
            if times:
                if os.path.exists(f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.pkl"):
                    cur_path = f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.pkl"
                else:
                    cur_path = f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.json"
            else:
                if os.path.exists(f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.pkl"):
                    cur_path = f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.pkl"
                else:
                    cur_path = f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.json"
            print(cur_path)
            p = pool.apply_async(task, args=(task_id, cur_path))
            processs.append(p)
        
        for each in processs:
            cur_results, cur_see_pass_num, cur_simple_pass_num_dict, cur_solution_dict = each.get()
            for task_id in cur_results:
                total_results[task_id] = cur_results[task_id]
            for task_id in cur_see_pass_num:
                total_see_pass_num[task_id] = cur_see_pass_num[task_id]
            for task_id in cur_simple_pass_num_dict:
                total_simple_pass_num_dict[task_id] = cur_simple_pass_num_dict[task_id]
            for task_id in cur_solution_dict:
                total_solution_dict[task_id] = cur_solution_dict[task_id]


        logging.basicConfig(
        format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        )
        logger = logging.getLogger(__name__)
        data = evaluate_with_test_code(merge_completion_and_prompt(total_solution_dict, dataset_dict, isplus), timeout=1, logger=logger)
        if data_type == 'human_eval' and isplus:
            dump_pickle(data, f"../runtime/human_eval_plus/{se_path}")
        else:
            dump_pickle(data, f"../runtime/{data_type}/{se_path}")

    total_results = {}
    total_see_pass_num = {}
    total_simple_pass_num_dict = {}
    total_solution_dict = {}
    


    pool = multiprocessing.Pool(processes=200)
    processs = []
    
    for task_id in tqdm(dataset_dict):

        if times:
            if os.path.exists(f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.pkl"):
                cur_path = f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.pkl"
            else:
                cur_path = f"../runtime/{data_type}/total_results/{times}/{task_id.replace('/', '_')}.json"
        else:
            if os.path.exists(f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.pkl"):
                cur_path = f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.pkl"
            else:
                cur_path = f"../runtime/{data_type}/total_results/{task_id.replace('/', '_')}.json"

        p = pool.apply_async(task, args=(task_id, cur_path))
        processs.append(p)
    
    for each in processs:
        cur_results, cur_see_pass_num, cur_simple_pass_num_dict, cur_solution_dict = each.get()
        for task_id in cur_results:
            total_results[task_id] = cur_results[task_id]
        for task_id in cur_see_pass_num:
            total_see_pass_num[task_id] = cur_see_pass_num[task_id]
        for task_id in cur_simple_pass_num_dict:
            total_simple_pass_num_dict[task_id] = cur_simple_pass_num_dict[task_id]
        for task_id in cur_solution_dict:
            total_solution_dict[task_id] = cur_solution_dict[task_id]




    ispassed = {}
    for dt in data:
        if dt["task_id"] not in ispassed:
            ispassed[dt["task_id"]] = {}
        ispassed[dt["task_id"]][dt["solution_id"]] = dt["passed"]
    task_ids = list(ispassed.keys())
    

    pass_at_k = {}
    pass_1_tasks = []
    for k in [1,2,5]:
        zero_num = []
        totals = []
        corrects = []
        for task_id in task_ids:
            if len(total_simple_pass_num_dict[task_id]) == 0:
                zero_num.append(task_id)
                totals.append(k)
                corrects.append(0)
                continue
            hat_n = k if k <= len(total_simple_pass_num_dict[task_id]) else len(total_simple_pass_num_dict[task_id])
            # print(task_id, total_simple_pass_num_dict[task_id])
            min_score = total_simple_pass_num_dict[task_id][hat_n-1][1]
            while hat_n < len(total_simple_pass_num_dict[task_id]):
                if total_simple_pass_num_dict[task_id][hat_n][1] != min_score:
                    break
                hat_n += 1
            assert total_simple_pass_num_dict[task_id][hat_n-1][1] == min_score
            totals.append(hat_n)
            corrects.append(sum([ispassed[task_id][x[0]] == 1 for x in total_simple_pass_num_dict[task_id][:hat_n]]))
            if k == 1 and corrects[-1] > 0:
                pass_1_tasks.append(task_id)
        cur_output = _estimate_pass_at_k(totals, corrects, k)
        print(len(cur_output), sum(cur_output == 0))
        pass_at_k[f"pass@{k}"] = round(_estimate_pass_at_k(totals, corrects, k).mean() * 100 , 2)
        print('zero num', len(zero_num), zero_num)
    if 'o1' in times:
        if data_type == 'human_eval' and isplus:
            data_type = 'human_eval_plus'
    if times:
        json.dump(pass_1_tasks, open(f"../runtime/{data_type}/pass_1_tasks_our_{times}.json", 'w'), indent=4)
    else:
        json.dump(pass_1_tasks, open(f"../runtime/{data_type}/pass_1_tasks_our.json", 'w'), indent=4)
    return pass_at_k


if __name__ == "__main__":
    consider_task = sys.argv[1]
    gen_model = sys.argv[2]
    type = sys.argv[3]
    isplus = sys.argv[4]
    if isplus == 'true':
        isplus = True
    else:
        isplus = False
    times = f'{consider_task}_gpt{gen_model}'
    if times == 'none':
        times = ""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    """pass@k"""
    consider_tasks = [(consider_task, isplus)]

    for data_type, isplus in consider_tasks:
        pass_origin = [str(x) for x in original_pass_at_k(data_type, isplus).values()]
        pass_tophatn = [str(x) for x in sorted_pass_at_k_top_hat_n(data_type, isplus, times).values()]

        print(f"{data_type}{'+' if isplus else ''} {' & '.join(pass_origin)} \\\\")
        print(f"{data_type}{'+' if isplus else ''} {' & '.join(pass_tophatn)} \\\\")