import argparse
import os
import sys
import json
import csv
from copy import deepcopy
from tqdm import tqdm
import random
import math
import Levenshtein
random.seed(111)

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-proportion", type=float, default=1.0)
    parser.add_argument("--val-ids-path")
    parser.add_argument("--problems-json")
    parser.add_argument("--num-attempts", type=int, default=None)
    parser.add_argument("--unique", action="store_true")
    parser.add_argument("--revision-threshold", type=float, default=0.5)
    parser.add_argument("--correction-threshold", type=float, default=0.7)
    parser.add_argument("--edit-threshold", type=float, default=1.0)
    parser.add_argument("--revision-with-error", action="store_true")
    parser.add_argument("--error-with-execution", action="store_true")
    parser.add_argument("--only-revision", action="store_true")
    parser.add_argument("--summary-prefix", default="sanjay")
    parser.add_argument("--trace-limit", type=int, default=100)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--part-num", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.system('mkdir \"'+args.output_dir+'\"')
    with open(args.problems_json) as f:
        problems = json.load(f)
    train_fieldnames = ['images','steps','prob_path','examples','context','prompts','programs']
    items = []
    # fname_list = sorted([fname for fname in os.listdir(args.input_dir) if fname[-5:] == '.json' and fname.split('-')[0] == args.summary_prefix])
    fname_list = [fname for fname in os.listdir(args.input_dir) if fname[-5:] == '.json' and fname.split('-')[0] == args.summary_prefix]
    fname_list = fname_list[args.part_num * len(fname_list) // args.num_parts:(args.part_num + 1) * len(fname_list) // args.num_parts]
    for fname in tqdm(fname_list):
        sys.stdout.flush()
        with open(os.path.join(args.input_dir, fname)) as f:
            data = json.load(f)
        saved_programs = set()
        limit = len(data['rewards']) if args.num_attempts is None else args.num_attempts
        reward_list = data['rewards'][:args.num_attempts]
        code_list = data['codes'][:args.num_attempts]
        prompt_list = data['prompts'][:args.num_attempts]
        if args.unique:
            good_indices = [i for i in range(len(data['rewards'])) if float(data['rewards'][i]) > 0.99]
            if len(good_indices) == 0:
                continue
            index = good_indices[0]
            reward_list = [data['rewards'][index]]
            code_list = [data['codes'][index]]
            prompt_list = [data['prompts'][index]]
        problem_path = problems[int(fname.split('.')[-2].split('-')[-1])]
        for reward, code, prompt in zip(reward_list, code_list, prompt_list):
            if float(reward) < 0.99:
                continue
            if code in saved_programs:
                continue
            if not args.only_revision:
                items.append({
                    'images': str(fname.split('.')[-2].split('-')[-1]),
                    'steps': str(len(saved_programs)),
                    'context': prompt.split('\nPROBLEM 3: ')[0],
                    'examples': prompt.split('\nPROBLEM 3: ')[1]+'```'+code+'```',
                    'prompts': prompt.split('\nPROBLEM 3: ')[1], # [prompt.index('\nPROBLEM 3'):],
                    'programs': '```'+code+'```',
                    'prob_path': problem_path
                })
            saved_programs.add(code)
        saved_programs = list(saved_programs)
        if args.revision_threshold < 0.99:
            if args.revision_with_error:
                program_pairs = set()
                with open(os.path.join(args.input_dir, 'testresults-'+fname.split('-')[1])) as f:
                    test_results = json.load(f)
                with open(os.path.join(problem_path, 'input_output.json')) as f:
                    inputs_outputs = json.load(f)
                num_test_cases = len(inputs_outputs['inputs'])
                index = 0
                print(fname)
                for reward, incorrect, prompt, testres in zip(data['rewards'], data['codes'], data['prompts'], test_results):
                    print(index, incorrect)
                    if float(reward) < args.revision_threshold and float(reward) < 1:
                        prefix = prompt.split('\nPROBLEM 3: ')[1]
                        traces_dir = os.path.join(args.input_dir, 'traces-'+fname.split('.')[-2].split('-')[-1]+'-'+str(index))
                        traces_files = list(os.listdir(traces_dir))
                        test_case_indices = [int(fname.split('.')[0].split('input')[1]) for fname in os.listdir(traces_dir) if 'input' in fname]
                        for index2 in range(len(data['codes'])):
                            if float(data['rewards'][index2]) >= max(args.correction_threshold, float(reward)):
                                correct_code = data['codes'][index2]
                                if (correct_code, incorrect) not in program_pairs:
                                    dist = Levenshtein.distance(correct_code, incorrect)
                                    if dist < args.edit_threshold:
                                        new_prompt = prefix+f'```{incorrect}```'
                                        if traces_files == ['compile.txt']:
                                            with open(os.path.join(traces_dir, traces_files[0])) as f:
                                                error_text = f.readlines()[0].strip()
                                            new_prompt += '\nCOMPILATION ERROR:\n'+error_text+'\n'
                                        elif len(test_case_indices) > 0:
                                            # print(traces_dir, list(os.listdir(traces_dir)))
                                            eligible_test_cases = [i 
                                                for i in test_case_indices 
                                                if min(len(test_results[index]), len(test_results[index2])) > i
                                                and test_results[index2][i] > test_results[index][i]
                                            ]
                                            if len(eligible_test_cases) == 0:
                                                continue
                                            test_case_index = random.choice(
                                                eligible_test_cases
                                            )
                                            inp = inputs_outputs['inputs'][test_case_index]
                                            out = inputs_outputs['outputs'][test_case_index]
                                            inp = ', '.join([str(val) for val in inp])
                                            if hasattr(out, 'len'):
                                                out = ', '.join([str(val) for val in out])
                                            else:
                                                out = str(out)
                                            new_prompt += '\nTEST CASE:'
                                            new_prompt += '\nInput: '+inp+'\nExpected Output: '+out+'\n'
                                            with open(os.path.join(traces_dir, 'input'+str(test_case_index)+'.txt')) as f:
                                                trace = f.readlines()
                                            if args.error_with_execution:
                                                new_prompt += 'TRACE:\n'
                                                print(os.path.join(traces_dir, 'input'+str(test_case_index)+'.txt'))
                                                if len(trace) > args.trace_limit:
                                                    continue
                                                code_lines = incorrect.split('\n')
                                                for line in trace:
                                                    parts = line.split('|')
                                                    if 'Event: line' in parts[0]:
                                                        line_num = int(parts[1].split('Line')[1].split(':')[0].strip())
                                                        print(code_lines, line_num)
                                                        new_prompt += 'Line '+code_lines[line_num] + ' | ' + parts[-1].strip()+'\n'
                                            if os.path.exists(os.path.join(traces_dir, 'error'+str(test_case_index)+'.txt')):
                                                with open(os.path.join(traces_dir, 'error'+str(test_case_index)+'.txt')) as f:
                                                    new_prompt += f.readlines()[0].strip()+'\n'
                                            else:
                                                if len(trace) == 0 or len(trace[-1].strip()) == 0:
                                                    new_prompt += 'TIMEOUT'
                                                else:
                                                    new_prompt += 'Output: '+trace[-1].strip()+'\n'
                                        if len(test_case_indices) > 0 and len((new_prompt+correct_code).split()) < 3000:
                                            new_prompt += 'CORRECTED ANSWER:\n'
                                            items.append({
                                                'images': 'rev-'+str(fname.split('.')[-2].split('-')[-1]),
                                                'steps': str(len(saved_programs)+len(program_pairs)),
                                                'context': prompt.split('\nPROBLEM: ')[0],
                                                'examples': new_prompt+'```'+correct_code+'```',
                                                'prompts': new_prompt,
                                                'programs': '```'+correct_code+'```',
                                                'prob_path': problem_path
                                            })
                                            program_pairs.add((correct_code, incorrect))
                    index += 1
            else:
                incorrect = None
                correct = None
                prefix = None
                for reward, code, prompt in zip(data['rewards'], data['codes'], data['prompts']):
                    if float(reward) >= args.revision_threshold and float(reward) < 1:
                        incorrect = code
                        prefix = prompt.split('\nPROBLEM 3: ')[1].replace('Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```', 'Fix the given program, which is slightly incorrect')
                    if float(reward) > 0.99:
                        correct = code
                if incorrect is not None and correct is not None:
                    full = prefix+f'\nWRONG CODE:\n```{incorrect}```\nCORRECT CODE:\n```{correct}```'
                    items.append({
                        'images': str(fname.split('.')[-2].split('-')[-1]),
                        'steps': str(-1),
                        'context': '',
                        'examples': full,
                        'prompts': prefix+f'\nWRONG CODE:\n```{incorrect}```\nCORRECT CODE:\n',
                        'programs': '```'+code+'```',
                        'prob_path': problem_path
                    })
    ids = set([item['images'] for item in items])
    if args.val_ids_path is not None:
        with open(args.val_ids_path) as f:
            reader = csv.DictReader(f)
            prev_val = list(reader)
        val_ids = set([item['images'] for item in prev_val])
        train_ids = set([ident for ident in ids if ident not in val_ids])
        present_ids = set([item['images'] for item in items])
        for item in prev_val:
            if item['images'] not in present_ids:
                for key in items[0]:
                    if key not in item:
                        item[key] = ''
                items.append(item)
    else:
        train_ids = set(random.sample(list(ids), int(math.ceil(len(ids)*args.train_proportion))))
        val_ids = set([ident for ident in ids if ident not in train_ids])
    train_items = [item for item in items if item['images'] in train_ids]
    val_items = [item for item in items if item['images'] in val_ids]
    with open(os.path.join(args.output_dir, 'train.csv'), 'w') as fout:
        writer = csv.DictWriter(fout, fieldnames=train_fieldnames)
        writer.writeheader()
        for item in train_items:
            writer.writerow(item)
        # writer.close()
    val_fieldnames = ['images','steps','prob_path','context', 'prompts','examples']
    with open(os.path.join(args.output_dir, 'val.csv'), 'w') as fout:
        writer = csv.DictWriter(fout, fieldnames=val_fieldnames)
        writer.writeheader()
        ids = set()
        for item in val_items:
            if item['images'] in ids:
                continue
            ids.add(item['images'])
            if int(item['steps']) >= 0:
                writer.writerow({key: item[key] for key in val_fieldnames})
        # writer.close()
