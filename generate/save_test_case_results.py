import argparse
import os
import json
import sys
from tqdm import tqdm
sys.path.append('../')
sys.path.append('../eval')
from eval.compute_reward import compute_reward, check_correctness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--problems-path")
    parser.add_argument("--prefix", default="sanjay")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    with open(args.problems_path) as f:
        problems = json.load(f)
    file_list = [fname for fname in os.listdir(args.input_dir) if fname[-5:] == '.json' and fname.split('-')[0] == args.prefix]
    end = len(file_list)
    if args.end is not None:
        end = args.end
    # print(file_list[:end], len(file_list[:end]))
    file_list = file_list[args.start:end]
    for fname in tqdm(file_list):
        if fname[-5:] == '.json' and fname.split('-')[0] == args.prefix:
            with open(os.path.join(args.input_dir, fname)) as f:
                res = json.load(f)
            index = int(fname.split('-')[1].split('.')[0])
            problem_path = problems[index]
            all_test_results = []
            print(fname, len(res['codes']))
            count = 0
            for code, reward in zip(res['codes'], res['rewards']):
                # if count != 16:
                #     count += 1
                #     continue
                trace_dir = os.path.join(args.input_dir, 'traces-'+str(index)+'-'+str(count))
                os.system('rm -rf \"'+trace_dir+'\"')
                os.system('mkdir \"'+trace_dir+'\"')
                test_results = check_correctness(problem_path, code, mode='train', public_test_cases='0', overfit=True, trace_path=trace_dir)
                # print(code)
                all_test_results.append(test_results)
                count += 1
            with open(os.path.join(args.input_dir, fname.replace(args.prefix, "testresults")), 'w') as fout:
                json.dump(all_test_results, fout)
