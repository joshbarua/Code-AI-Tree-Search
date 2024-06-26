import copy
import json
import os
import warnings
from abc import abstractmethod, ABC
from collections import OrderedDict
from types import SimpleNamespace
import random
import time

from eval.compute_reward import compute_reward
from eval.generate_gpt_codes import get_output_str_from_state_for_apps


class ProgramEnv(ABC):
    """
    Code generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Reward: pass rate of the program (on the training set in training, and on the test set in testing).
    """
    def __init__(self, terminal_token, horizon):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon

        # state -> reward
        # we may need to retrieve the states (programs) in the order they were saved, so use OrderedDict
        self.cached_reward = OrderedDict()
        self.cached_time = OrderedDict()

    def transition(self, s, a, is_model_dynamic=True):
        next_state = s + [a]

        if a == self.terminal_token or len(next_state) == self.horizon:
            # either the program finishes, or the state reaches the maximum length
            done = True
        else:
            done = False

        if done:
            reward = self.get_reward(next_state)
        else:
            reward = 0  # no intermediate reward

        return next_state, reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)

        return self.state, reward, done, {}

    @abstractmethod
    def get_reward(self, s, mode='train'):
        """
        This needs to be defined for each dataset
        """
        pass

    def convert_state_to_program(self, s):
        """
        The state may be different from the program. This converts it back to executable program.
        """
        return s

    def equality_operator(self, s1, s2):
        return s1 == s2

    def get_complete_programs(self):
        """
        Return the list of complete programs reached so far.
        This can be found from the list of cached rewards.
        """
        return list(map(lambda x: list(x), self.cached_reward.keys()))


class APPSProgramEnv(ProgramEnv):
    """
    Code generation environment for APPS dataset.
    """
    def __init__(self, prob_path, tokenizer, model_name, horizon, public_test_cases=None, in_context_problems=None, overfit=False, min_generation_length=500, strip_prompt_whitespace=False, extra_examples=[], num_in_context_examples=None, programs_to_repair=None):
        self.prob_path = prob_path
        self.tokenizer = tokenizer
        self.model = model_name
        self.public_test_cases = public_test_cases
        self.in_context_problems = in_context_problems
        self.num_in_context_examples = num_in_context_examples
        self.overfit = overfit
        self.horizon = horizon
        self.min_generation_length = min_generation_length
        self.strip_prompt_whitespace = strip_prompt_whitespace
        self.extra_examples = extra_examples
        self.programs_to_repair = programs_to_repair

        # code from generate_gpt_codes that generate paths for all essential files
        public_test_case_path = os.path.join(prob_path, "public_input_output.json")
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")

        if not os.path.exists(starter_path):
            starter_path = None
        if not os.path.exists(test_case_path):
            raise ValueError("input_output.json missing so can't do testing. Invalid ProgramEnv.")
        if not os.path.exists(prompt_path):
            raise Exception("question.json missing. Invalid ProgramEnv.")
        if public_test_cases == 'desc' and not os.path.exists(public_test_case_path):
            raise Exception('using public test cases in problem description, but public test cases missing.')

        from eval.generate_gpt_codes import generate_apps_prompt
        # generate prompt to encode question text and an "ANSWER" prompt to the state
        # no need to load the full arglist here, it only needs to check the value of peeking (using default value 0.0 here)
        gpt_args = SimpleNamespace(peeking=0.0)
        state, _ = generate_apps_prompt(gpt_args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)

        self.init_prompt = copy.copy(state)
        if self.in_context_problems: # add in-context examples to prompt
            self.starter_path = starter_path
        state = self.set_in_context_examples()

        self.state = self.tokenizer.encode(state)
        #terminal_token = self.tokenizer.encode('<|endoftext|>')[0]
        terminal_token = self.tokenizer.encode('</s>')[0]

        super(APPSProgramEnv, self).__init__(terminal_token=terminal_token, horizon=horizon)

    def get_repair_prompts(self):
        if self.programs_to_repair is not None and len(self.programs_to_repair) > 0:
            prompts = []
            for program_to_repair in self.programs_to_repair:
                for correction_prompt in program_to_repair['prompts']:
                    prompt = "Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:"+copy.copy(self.init_prompt)+"```"+program_to_repair['code']+"```"+correction_prompt+'CORRECTED ANSWER:'
                    prompts.append(prompt)
            return prompts
        return None

    def set_in_context_examples(self):
        if not self.in_context_problems:
            return "Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:"+copy.copy(self.init_prompt)
        from eval.generate_gpt_codes import generate_apps_prompt
        gpt_args = SimpleNamespace(peeking=1, peek_frac=1) # want to peek at solutions
        attempts = 0
        while True:
            in_context_sample = random.sample(list(range(len(self.in_context_problems)+len(self.extra_examples))), self.num_in_context_examples)
            # in_context_sample = random.sample(self.in_context_problems, 2)
            in_context_prompt = f"You are an expert python programmer tasked with solving {len(in_context_sample)+1} problems. Wrap your solutions inside a code block.\n"
            for i,index in enumerate(in_context_sample):
                if index < len(self.in_context_problems):
                    path = self.in_context_problems[index]
                    test_case_path = os.path.join(path, "input_output.json")
                    prompt_path = os.path.join(path, "question.txt")
                    if os.path.exists(f"{path}/starter_code.py"):
                        starter_path = os.path.join(path, "starter_code.py")
                    else:
                        starter_path = None
                    solutions_path = os.path.join(path, "solutions.json")
                    example, _ = generate_apps_prompt(gpt_args, test_case_path, prompt_path, solutions_path, self.tokenizer, starter_path)
                    in_context_prompt += f"PROBLEM {i+1}: Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:{example}\n"
                else:
                    prompt = self.extra_examples[index-len(self.in_context_problems)][0]
                    program = random.choice(self.extra_examples[index-len(self.in_context_problems)][1])[3:-3]
                    in_context_prompt += f"PROBLEM {i+1}: "+prompt+'```'+program+'```\n'
            state = copy.copy(self.init_prompt)
            state = f"{in_context_prompt}PROBLEM {len(in_context_sample)+1}: Write a python code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:{state}"
            if self.strip_prompt_whitespace:
                state = state.strip()
            prompt_length = len(self.tokenizer.encode(state))
            if prompt_length < self.horizon - self.min_generation_length:
                return state
            attempts += 1
            if attempts >= 10:
                break                
            # print('prompt length', prompt_length)
        return state

    def convert_state_to_program(self, s, prompt=None):
        s = self.tokenizer.decode(s, skip_special_tokens=True)
        if self.in_context_problems:
            if prompt is not None:
                s = s[len(prompt):]
            # print('output')
            # print(s)
            if "PROBLEM" in s:
                s = s[:s.find("PROBLEM")]
        return get_output_str_from_state_for_apps(s)

    def get_canonical_state(self):
        raise NotImplementedError()

    def get_reward(self, s, start_time=None, mode='train'):
        """
        Returns:
            The reward of program in s.
        """
        if s is None:
            return 0

        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            return self.cached_reward[tuple(s)]

        output_str = self.convert_state_to_program(s)
        reward = compute_reward(self.prob_path, output_str, mode=mode, public_test_cases=self.public_test_cases, overfit=self.overfit)

        if mode == 'train':
            if start_time:
                end_time = time.time()
                elapsed_time = (end_time-start_time)
                print("The time of execution of above program is :",elapsed_time, "s")
                self.cached_time[tuple(s)] = elapsed_time
            self.cached_reward[tuple(s)] = reward

        return reward
