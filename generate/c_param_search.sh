#!/bin/bash

for c in 4 8 12;
do
    python synthesis_exp.py -s 0 -e 5000 --alg mcts --rollout 16 --prefix t-c$c- --test-loc ../data_split/200prob_train_introductory.json --horizon 4096 --rerun --in_context_examples ../data_split/10prob_train_introductory.json --ucb-constant $c --save ./results_c$c --max-sample-times 4096
done