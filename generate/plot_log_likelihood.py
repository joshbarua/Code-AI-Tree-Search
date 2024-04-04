import argparse
import os
import json
from transformers import LlamaForCausalLM, CodeLlamaTokenizer, BitsAndBytesConfig
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--data_loc", type=str, required=True)
parser.add_argument("--file_prefix", type=str, required=True)
args = parser.parse_args()

compute_dtype = getattr(torch, "bfloat16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = LlamaForCausalLM.from_pretrained(args.model, quantization_config=quant_config, device_map="auto")
tokenizer = CodeLlamaTokenizer.from_pretrained(args.model)
tokenizer.pad_token_id = 0

loglikelihoods = []
rewards = []

for filename in os.listdir(args.data_loc):
    if args.file_prefix not in filename:
        continue 
    
    with open(f"{args.data_loc}/{filename}") as json_file:
        data = json.load(json_file)
        
    prompts = data["prompts"]
    
    for i in range(data["sample times"]):
        combined_prompts = []
        prompt = prompts[i]
        task_desc = prompt[prompt.find("PROBLEM 3:"):]
        task_desc = task_desc.replace('ANSWER:', 'QUESTION:')
        task_desc = task_desc.split('QUESTION:')[1]
        context = prompt[:prompt.find("PROBLEM 3:")]
        context = context.replace('ANSWER:', 'QUESTION:')
        context = context.split('QUESTION:')
        for i in [1,3]: # for 2 in-context examples
            in_context_example = context[i]
            combined_prompt = f"{in_context_example}\n{task_desc}"
            combined_prompts.append(combined_prompt)
        inputs = tokenizer(combined_prompts, padding=True, return_tensors="pt")
        logits = model(**inputs)["logits"]
        logprobs = torch.max(logits, dim=2)[0]
        mean_logprobs = torch.mean(logprobs, dim=1)
        print(mean_logprobs.shape)
        5/0
        rewards.append(data["rewards"][i])
        print(loglikelihoods)
        print(rewards)
        5/0
    
