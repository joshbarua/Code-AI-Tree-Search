#!/bin/bash

for k in 4 5 6;
do
    python synthesis_exp.py -s 0 -e 5000 --alg mcts --rollout 16 --prefix t-k$k- --test-loc ../data_split/200prob_train_introductory.json --horizon 4096 --rerun --in_context_examples ../data_split/10prob_train_introductory.json --width $k --save ./results_k$k --max-sample-times 4096 --public-cases 0 --overfit
done