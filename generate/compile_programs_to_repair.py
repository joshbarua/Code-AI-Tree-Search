import argparse
import os
from tqdm import tqdm
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    parser.add_argument("--output-path")
    parser.add_argument("--input-prefix", default="sanjay")
    parser.add_argument("--problems-path")
    parser.add_argument("--trace-limit", type=int, default=100)
    parser.add_argument("--program-limit", type=int, default=None)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--part-num", type=int, default=0)
    args = parser.parse_args()
    output = {}
    with open(args.problems_path) as f:
        problems = json.load(f)
    fnames = [fname for fname in os.listdir(args.input_dir) if  fname[-5:] == '.json' and fname.split("-")[0] == args.input_prefix]
    fnames = fnames[args.part_num*len(fnames)//args.num_parts:(args.part_num+1)*len(fnames)//args.num_parts]
    for fname in tqdm(fnames):
        with open(os.path.join(args.input_dir, fname)) as f:
            res = json.load(f)
        if max(res['rewards']) > 0.99:
            continue
        indices = sorted(list(range(len(res['rewards']))), key=lambda x: -res['rewards'][x])
        if args.program_limit is not None:
            indices = [i for i in indices if i < args.program_limit]
        with open(os.path.join(args.input_dir, 'testresults-'+fname.split('-')[1])) as f:
            test_results = json.load(f)
        problem_path = problems[int(fname.split('-')[1].split('.')[0])]
        with open(os.path.join(problem_path, 'input_output.json')) as f:
            inputs_outputs = json.load(f)
        output[problem_path] = []
        for i in indices:
            traces_dir = os.path.join(args.input_dir, 'traces-'+fname.split('.')[-2].split('-')[-1]+'-'+str(i))
            traces_files = list(os.listdir(traces_dir))
            test_case_indices = [int(fname.split('.')[0].split('input')[1]) for fname in os.listdir(traces_dir) if 'input' in fname]
            prompts = []
            if traces_files == ['compile.txt']:
                with open(os.path.join(traces_dir, traces_files[0])) as f:
                    error_text = f.readlines()[0].strip()
                    prompts.append('\nCOMPILATION ERROR:\n'+error_text+'\n')
            elif len(traces_files) > 0:
                for t in test_case_indices:
                    prompt = ""
                    inp = inputs_outputs['inputs'][t]
                    out = inputs_outputs['outputs'][t]
                    inp = ', '.join([str(val) for val in inp])
                    if hasattr(out, 'len'):
                        out = ', '.join([str(val) for val in out])
                    else:
                        out = str(out)
                    prompt += '\nTEST CASE:'
                    prompt += '\nInput: '+inp+'\nExpected Output: '+out+'\n'
                    with open(os.path.join(traces_dir, 'input'+str(t)+'.txt')) as f:
                        trace = f.readlines()
                    prompt += 'TRACE:\n'
                    # print(os.path.join(traces_dir, 'input'+str(t)+'.txt'))
                    if len(trace) > args.trace_limit:
                        continue
                    code_lines = res['codes'][i].split('\n')
                    for line in trace:
                        parts = line.split('|')
                        if 'Event: line' in parts[0]:
                            line_num = int(parts[1].split('Line')[1].split(':')[0].strip())
                            # print(code_lines, line_num)
                            prompt += 'Line '+code_lines[line_num] + ' | ' + parts[-1].strip()+'\n'
                    if os.path.exists(os.path.join(traces_dir, 'error'+str(t)+'.txt')):
                        with open(os.path.join(traces_dir, 'error'+str(t)+'.txt')) as f:
                            prompt += f.readlines()[0].strip()+'\n'
                    else:
                        if len(trace) == 0 or len(trace[-1].strip()) == 0:
                            prompt += 'TIMEOUT'
                        else:
                            prompt += 'Output: '+trace[-1].strip()+'\n'
                    prompts.append(prompt)
            if len(prompts) > 0:
                output[problem_path].append({
                    'code': res['codes'][i],
                    'prompts': prompts,
                    'reward': res['rewards'][i]
                })
    with open(args.output_path, 'w') as fout:
        json.dump(output, fout)
