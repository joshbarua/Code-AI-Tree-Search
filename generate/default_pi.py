import time
import warnings
from abc import abstractmethod

import torch
import numpy as np
import together
import requests

from transformer_utils.cache import GPTTopKCache, GPTSeqCache
import vllm

together.api_key = "9bc47a9e93c37fb4c1960da75d1fd6e16edddd348da059d2cc8286ea2a44d342"

class DefaultPolicyHeuristic:
    def __init__(self, k, horizon, env):
        self.k = k
        self.horizon = horizon
        self.env = env
        self.sample_times = 0
        self.time_stamps = [] # time stamp when a new sample is generated

    @abstractmethod
    def get_predict_sequence(self, state, horizon=None):
        pass

    @abstractmethod
    def get_top_k_predict(self, state):
        pass

    def clean_up(self, new_state):
        # implement this if need to do anything after each token is generated
        pass


class APPSHeuristic(DefaultPolicyHeuristic):
    def __init__(self,
                 tokenizer,
                 model,
                 k,
                 num_beams,
                 test_all_beams,
                 horizon,
                 device,
                 env,
                 value_model=None,
                 new_token_num=None,
                 use_seq_cache=False, # disable all caching by default
                 use_prompt_cache=False,
                 top_k_cache_steps=0,
                 ts_mode='best',
                 debug=False,
                 in_context_examples=False,
                 resample_incontext=False,
                 parallel_context_window=False):
        super(APPSHeuristic, self).__init__(k=k, horizon=horizon, env=env)

        self.tokenizer = tokenizer
        self.k = k
        self.num_beams = num_beams
        self.test_all_beams = test_all_beams
        self.horizon = horizon
        self.new_token_num = new_token_num
        self.device = device
        self.env = env

        self.use_seq_cache = use_seq_cache
        self.use_prompt_cache = use_prompt_cache # todo
        self.top_k_cache_steps = top_k_cache_steps
        self.ts_mode = ts_mode

        self.debug = debug
        self.in_context_examples = in_context_examples
        self.resample_incontext = resample_incontext
        self.parallel_context_window = parallel_context_window

        self.model = model
        self.value_model = value_model

        self.use_value = (self.value_model is not None)

        if self.ts_mode == 'sample' and self.use_seq_cache:
            warnings.warn("Cannot use sequence caching in sample mode, disabling it.")
            self.use_seq_cache = False

        if self.use_value and self.new_token_num is None:
            warnings.warn("Using a value function but not setting a shorter planning horizon (args.new_token_num)."
                          "Why using a value function?")

        #self.model.to(device)
        if self.use_value:
            self.value_model.to(device)

        if device == torch.device('cuda'):
            if hasattr(self.model, 'parallelize'):
                self.model.parallelize()
            if self.value_model is not None and hasattr(self.model, 'parallelize'):
                self.value_model.parallelize()

        self.top_k_cache = GPTTopKCache(k, cache_steps=top_k_cache_steps, tokenizer=tokenizer)
        self.seq_cache = GPTSeqCache()
        self.prompt_key_values = None

        self.terminal_token = self.env.terminal_token

    def get_short_horizon_sequence(self, state):
        """
        Returns:
            predicted sequence, but only with up to self.new_token_num new tokens.
            This uses self.get_predict_sequence.
        """
        # add length of prompt and existing program
        horizon = len(state) + self.new_token_num
        # don't exceed the length of Transformer input
        horizon = min(horizon, self.horizon)
        if horizon > len(state):
            return self.get_predict_sequence(state, horizon=horizon)
        else:
            return state

    def get_predict_sequence(self, state, horizon=None, return_all=False):
        """
        Args:
            horizon: return a new sequence with this extra length
        Returns:
            Get the most likely sequence starting from state.
        """
        with torch.no_grad():
            encoded_ids = state # as a list
            input_ids = [torch.LongTensor(encoded_ids).to(self.device)]
            inputs = [self.tokenizer.decode(input_ids[0], skip_special_tokens=True)]
            if self.env.programs_to_repair is not None:
                inputs = self.env.get_repair_prompts()
                if inputs is None:
                    inputs = [self.env.set_in_context_examples() for _ in range(self.num_beams)]
                inputs = inputs[:self.num_beams]
                input_ids = [torch.LongTensor(self.tokenizer.encode(input_str)).to(self.device) for input_str in inputs]
            elif self.resample_incontext:
                inputs = [self.env.set_in_context_examples() for _ in range(self.num_beams)]
                input_ids = [torch.LongTensor(self.tokenizer.encode(input_str)).to(self.device) for input_str in inputs]
            self.env.prompts = inputs
            print(inputs[0])

            if self.use_seq_cache:
                output_ids = self.seq_cache.get(encoded_ids)
                if output_ids is not None:
                    print("CACHED PROGRAM", self.tokenizer.decode(output_ids))
                    print("found cached output ids")
                    return output_ids

            prompt_length = max([ids.numel() for ids in input_ids])
            if horizon is None:
                horizon = max(self.horizon, prompt_length+1)

            start_time = time.time()
            prompt_length = min([ids.numel() for ids in input_ids])
            sample_mode = (self.ts_mode == 'sample')

            max_length = max([ids.numel()+horizon-prompt_length for ids in input_ids])
            if self.parallel_context_window:
                assert self.num_beams == 1 or sample_mode
                preamble = inputs[0].split('PROBLEM 1: ')[0]
                examples = set()
                for input_str in inputs:
                    for j in range(self.env.num_in_context_examples):
                        examples.add(input_str.split('\nPROBLEM '+str(j+1)+': ')[1].split('PROBLEM '+str(j+2)+': ')[0])
                problem = '\nPROBLEM '+str(self.env.num_in_context_examples+1)+': '+inputs[0].split('\nPROBLEM '+str(self.env.num_in_context_examples+1)+': ')[1]
                contexts = [txt.split('\nPROBLEM '+str(self.env.num_in_context_examples+1)+': ')[0] for txt in inputs]

                model_output = [
                    self.model.pcw_generate(
                        contexts=contexts,
                        task_text=problem,
                        top_k=(0 if sample_mode else self.k),
                        temperature =(.6 if sample_mode else 0),
                        top_p=(.95 if sample_mode else 0),
                        num_beams=(1 if sample_mode else self.num_beams), # if sampling enabled, beam should always be 1
                        num_return_sequences=self.num_beams,
                        do_sample=sample_mode,
                        early_stopping=True,
                        return_dict_in_generate=True,
                        output_hidden_states=False,
                        output_attentions=False,
                        output_scores=True,
                        max_length=horizon,
                        use_cache=True, # huggingface default cache is always enabled
                        low_memory=(True if self.num_beams > 1 else False)
                    )
                    for _ in range(self.num_beams)
                ]
                input_lengths = [t.numel() for t in input_ids]
                if not self.resample_incontext:
                    input_lengths = [input_lengths[0] for _ in range(model_output)]
                sequences = [
                    model_output[i].sequences[0,input_lengths[i]:]
                    for i in range(model_output)
                ]
                sequences = torch.stack([
                    torch.cat((
                        sequence,
                        self.env.tokenizer.eos_token_id*torch.ones((max_length-sequence.numel())).to(sequence.device)
                    ), dim=0)
                    if sequence.numel() < max_length else sequence
                    for sequence in sequences
                ])
                scores = torch.Tensor([t.scores for t in model_output]).to(self.device)
            else:
                sampling_params = vllm.SamplingParams(
                    temperature=(0.6 if sample_mode else 0),
                    top_p=(0.95 if sample_mode else 1.0),
                    n=self.num_beams if not self.resample_incontext else 1,
                    use_beam_search=((not sample_mode) and self.num_beams > 1),
                    max_tokens=horizon-prompt_length,
                )
                model_output = self.model.generate(
                    prompts=inputs,
                    sampling_params=sampling_params,
                    prompt_token_ids=[ids.tolist() for ids in input_ids]
                )
                if not self.resample_incontext:
                    model_output = model_output[0]
                    sequences = torch.cat((
                        # input_ids[0].unsqueeze(0).repeat(len(model_output.outputs), 1),
                        torch.stack([
                            torch.cat((torch.LongTensor(output.token_ids), self.env.tokenizer.eos_token_id*torch.ones((horizon-len(output.token_ids)))), dim=0) if len(output.token_ids) < horizon else torch.LongTensor(output.token_ids)
                            for output in model_output.outputs
                        ]).to(self.device),
                    ), dim=1)
                    scores = torch.Tensor([
                        # output.cumulative_logprob.item() if isinstance(output.cumulative_logprob, torch.Tensor) else output.cumulative_logprob
                        output.cumulative_logprob
                        for output in model_output.outputs
                    ]).to(self.device)
                else:
                    sequences = torch.stack([
                        torch.cat((
                            # in_ids,
                            torch.LongTensor(output.outputs[0].token_ids).to(self.device),
                            self.env.tokenizer.eos_token_id*torch.ones((max_length-len(output.outputs[0].token_ids))).to(self.device)
                        ), dim=0)
                        if len(output.outputs[0].token_ids) < max_length else torch.cat((
                            # in_ids,
                            torch.LongTensor(output.outputs[0].token_ids).to(self.device),
                        ), dim=0)
                        for output, in_ids in zip(model_output, input_ids)
                    ]).to(self.device)
                    scores = torch.Tensor([
                        # output.cumulative_logprob.item() if isinstance(output.cumulative_logprob, torch.Tensor) else output.cumulative_logprob
                        output.outputs[0].cumulative_logprob
                        for output in model_output
                    ]).to(self.device)
            print(self.env.tokenizer.decode(sequences[0], skip_special_tokens=True))
            # print(model_outpt[0].outputs[0].token_ids)
            """while True:
                try:
                    model_output = together.Complete.create(
                        prompt = self.env.tokenizer.decode(input_ids[0,:], skip_special_tokens=True),
                        model = "togethercomputer/CodeLlama-7b-Python", 
                        max_tokens = horizon,
                        temperature = (.6 if sample_mode else 0),
                        top_k = (0 if sample_mode else self.k),
                        top_p = (0.95 if sample_mode else 0),
                        repetition_penalty = 1.1,
                        stop = ['```\n']
                    )
                    break
                except (together.error.RateLimitError, requests.exceptions.HTTPError) as e:
                    pass
            print(model_output['output']['choices'][0]['text'])"""

            """if self.top_k_cache_steps > 0:
                if hasattr(model_output, 'beam_indices'):
                    # beam search output
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores, beam_indices=model_output.beam_indices)
                else:
                    self.top_k_cache.add(input_ids, model_output.sequences, model_output.scores)"""

            if self.debug: print('generate sequence time: ' + str(time.time() - start_time))

            output_ids_list = sequences.tolist()
            # output_ids_list = [self.env.tokenizer(self.env.tokenizer.decode(input_ids[0,:], skip_special_tokens=True)+model_output['output']['choices'][i]['text'], return_tensors='pt')['input_ids'].view(-1) for i in range(len(model_output['output']['choices']))]

            if len(output_ids_list) > 1 and self.test_all_beams:
                # if got multiple output_ids using beam search, and going to test all beams (which takes more time)
                # pick the one that has the highest reward
                if self.debug:
                    for output_ids in output_ids_list:
                        print('==== generated program ====')
                        print(self.env.convert_state_to_program(output_ids))
                        print('===========================')
                cand_rewards = [self.env.get_reward(output_ids) for output_ids in output_ids_list]
                output_ids = output_ids_list[np.argmax(cand_rewards)]
            else:
                output_ids = output_ids_list[0]

            if self.use_seq_cache:
                self.seq_cache.add(encoded_ids, output_ids)

            self.sample_times += 1
            self.time_stamps.append(time.time())
            #print(output_ids)
            #print(self.tokenizer.convert_ids_to_tokens(output_ids))

            if self.debug:
                print('==== selected program ====')
                print(self.env.convert_state_to_program(output_ids))
                print('===========================')
                print('NUM TOKENS', len(output_ids)-prompt_length)

            if return_all:
                return output_ids_list
            return output_ids

    def get_value(self, state):
        with torch.no_grad():
            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)
            est_value = self.value_model(input_ids).logits.item()

            if self.debug:
                print(f"estimated value is {est_value}")

            return est_value

    def get_top_k_predict(self, state):
        """
        Returns:
            A list of k most likely tokens generate in state (descending in their scores)
            The probability of each action
        """
        with torch.no_grad():
            if self.top_k_cache_steps > 0:
                top_k_info = self.top_k_cache.get(state)
                if top_k_info is not None:
                    if self.debug: print('top-k cache hit')
                    return top_k_info

            encoded_ids = state
            input_ids = torch.LongTensor(encoded_ids).unsqueeze(0).to(self.device)

            start_time = time.time()

            """model_output = self.model.generate(
                input_ids,
                top_k=self.k,
                num_beams=self.num_beams,
                early_stopping=True,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )


            top_k_scores, top_k_tokens = torch.topk(model_output.scores[0][0], k=self.k, sorted=True)
            top_k_scores = torch.softmax(top_k_scores, dim=-1)"""
            sampling_params = vllm.SamplingParams(
                temperature=0,
                top_p=1.0,
                n=self.k,
                use_beam_search=True,
                max_tokens=1,
            )
            model_output = self.model.generate(
                prompts=self.env.tokenizer.decode(input_ids[0,:], skip_special_tokens=True),
                sampling_params=sampling_params,
                prompt_token_ids=[input_ids[0,:].tolist()]
            )[0]
            top_k_tokens = torch.LongTensor([
                output.token_ids for output in model_output.outputs
            ]).view(-1).to(self.device)
            top_k_scores = torch.Tensor([
                output.cumulative_logprob
                for output in model_output.outputs
            ]).to(self.device).exp()
            top_k_scores /= top_k_scores.sum().item()
            top_k_scores, indices = top_k_scores.sort(dim=-1, descending=True)
            top_k_tokens = top_k_tokens[indices.tolist()]

            if self.debug: print('generate top-k time: ' + str(time.time() - start_time))

            return top_k_tokens.tolist(), top_k_scores.tolist()

    def clean_up(self, new_state):
        if self.use_seq_cache:
            # clear hashed sequences that are not consistent with new_state
            self.seq_cache.clear(new_state)

        if self.top_k_cache_steps > 0:
            self.top_k_cache.clear(new_state)

