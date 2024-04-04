import argparse
import os
import json
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, BitsAndBytesConfig
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data_loc", type=str, required=True)
parser.add_argument("--file_prefix", type=str, default="sanjay-")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-parts", type=int, default=1)
parser.add_argument("--part-num", type=int, default=0)
parser.add_argument("--output-dir")
args = parser.parse_args()

compute_dtype = getattr(torch, "bfloat16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = LlamaForCausalLM.from_pretrained(args.model, quantization_config=quant_config, device_map="auto")
model = model.eval()
tokenizer = CodeLlamaTokenizer.from_pretrained(args.model)
fnames = [fname for fname in os.listdir(args.data_loc) if args.file_prefix in fname and fname[-5:] == '.json']
fnames = fnames[args.part_num*len(fnames) // args.num_parts:(args.part_num+1)*len(fnames) // args.num_parts]

examples = []
for filename in fnames:
    with open(f"{args.data_loc}/{filename}") as json_file:
        data = json.load(json_file)

    prompts = data["prompts"]
        
    for i in range(data["sample times"]):
        prompt = prompts[i]
        task_desc = prompt[prompt.find("PROBLEM 3:"):]
        task_desc = task_desc.replace('ANSWER:', 'QUESTION:')
        task_desc = task_desc.split('QUESTION:')[1]
        examples.append(prompt.split('PROBLEM 1:')[1].split('PROBLEM 2:')[0].strip())
        examples.append(prompt.split('PROBLEM 2:')[1].split('PROBLEM 3:')[0].strip())

examples = list(set(examples))
print('NUMBER OF IN CONTEXT EXAMPLES:', len(examples))
losses = []
rewards = []
tokenizer.pad_token = tokenizer.eos_token
for filename in tqdm(fnames):
    with open(f"{args.data_loc}/{filename}") as json_file:
        data = json.load(json_file)
        
    prompts = data["prompts"]
    task_desc = prompts[0].split("PROBLEM 3:")[1].strip()
    programs = data['codes'][:5]
    loss_lst = []
    with torch.no_grad():
        for i in range(0, len(programs), args.batch_size):
            # prompts = ["You are an expert python programmer tasked with solving 3 problems. Wrap your solutions inside a code block.\nPROBLEM 1: {ex}\nPROBLEM 2: "+task_desc for ex in examples[i:i+args.batch_size]]
            prompts = [task_desc[task_desc.find('QUESTION:'):].strip()+'\n```\n'+program.strip()+'\n```' for program in programs[i:i+args.batch_size]]
            print(filename)
            inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda:0')
            prompt_inputs = tokenizer([task_desc[task_desc.find('QUESTION:'):].strip()], return_tensors='pt')
            inputs['labels'] = inputs.input_ids
            inputs['labels'] = torch.where(
                inputs['attention_mask'] > 0,
                inputs['labels'],
                -100*torch.ones_like(inputs['labels'])
            )
            inputs['labels'][:,:prompt_inputs.input_ids.shape[-1]] = -100
            logits = model(**inputs).logits
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fn(logits[:,:-1,:].contiguous().view(-1, logits.shape[-1]), inputs['labels'][:,1:].contiguous().view(-1)).view(logits.shape[0], -1)
            lengths = inputs['attention_mask'][:,1:].sum(dim=1)
            loss = loss.sum(dim=1) / lengths
            loss_lst.append(loss.view(-1).cpu())
            break
    losses.append(torch.cat(loss_lst, dim=0).tolist())
    rewards.append(data['rewards'])
    # break
os.system('mkdir \"'+args.output_dir+'\"')
with open(os.path.join(args.output_dir, f"losses_5attempts_part{args.part_num}.json"), 'w') as f:
    json.dump(losses, f)
with open(os.path.join(args.output_dir, f"rewards_part{args.part_num}.json"), 'w') as f:
    json.dump(rewards, f)
