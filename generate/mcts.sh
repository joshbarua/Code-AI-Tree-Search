python synthesis_exp.py -s 0 -e 5000 --alg mcts --rollout 5 --prefix t- --test-loc ../data_split/test.json --horizon 4096 --save ./results_test --in_context_examples ../data_split/train_introductory.json --max-sample-times 4096 --rerun