import json
import os
import argparse
import sys
from transformers import AutoTokenizer
from types import SimpleNamespace
import csv
import random
sys.path.append('../')
sys.path.append('../eval/')
random.seed(111)
from eval.generate_gpt_codes import generate_apps_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-problems-path")
    parser.add_argument("--examples-csv-path")
    parser.add_argument("--base-model-path")
    parser.add_argument("--output-path")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    with open(args.initial_problems_path) as f:
        problems = json.load(f)
    examples = {}
    for prob_path in problems:
        problem_id = prob_path.split('/')[-1]
        # code from generate_gpt_codes that generate paths for all essential files
        public_test_case_path = os.path.join(prob_path, "public_input_output.json")
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")

        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(test_case_path):
            raise Exception("input_output.json missing so can't do testing. Invalid ProgramEnv.")
        if not os.path.exists(prompt_path):
            raise Exception("question.json missing. Invalid ProgramEnv.")

        # generate prompt to encode question text and an "ANSWER" prompt to the state
        # no need to load the full arglist here, it only needs to check the value of peeking (using default value 0.0 here)
        gpt_args = SimpleNamespace(peeking=0.0)
        problem_statement, _ = generate_apps_prompt(gpt_args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
        with open(solutions_path) as f:
            solutions = json.load(f)
        example = {
            "instructions": "Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:"+problem_statement,
            "programs": random.choice(solutions)
        }
        examples[problem_id] = example

    if args.examples_csv_path is not None:
        with open(args.examples_csv_path) as f:
            new_examples = csv.DictReader(f, fieldnames=['images','steps','gt_board','examples','context','prompts','programs'])
            for ex in new_examples:
                if ex["images"] in examples:
                    continue
                prompt = ex['prompts']
                program = ex['programs'][3:-3]
                example = {
                    "instructions": prompt,
                    "programs": program
                }
                examples[ex["images"]] = example
    with open(args.output_path, 'w') as fout:
        json.dump(examples, fout)
