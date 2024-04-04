# python /home/sanjayss/joshbarua/code-hexagons/finetune.py --base-model codellama/CodeLlama-7b-Python-hf --lr 3e-4 --train-csv-path round1_formatted_unique_probpath_csv/train.csv --val-csv-path round1_formatted_unique_probpath_csv/val.csv --shuffle --batch-size 8 --task apps --train-eval --loss-only-on-code --max-new-tokens 200 --vllm --checkpoint-path finetuned_models/merged_codellama7b_round1_lr3e-4_formatted_unique_shuffle_eval_lorar64_codeonlyloss_checkpoint-95 # _1example --examples-path round1_formatted_csv/examples.json --random-retrieval --apps
# TheBloke/CodeLlama-7B-Python-AWQ
python /home/sanjayss/joshbarua/code-hexagons/finetune.py --base-model TheBloke/CodeLlama-7B-Python-AWQ --lr 3e-4 --train-csv-path round1_formatted_unique_probpath_csv/train.csv --val-csv-path round1_formatted_probpath_redo_csv/val.csv --shuffle --batch-size 8 --task apps --train-eval --loss-only-on-code --max-new-tokens 200 --in-context --vllm --awq --balanced-sampling # --checkpoint-path finetuned_models/codellama7b_round1_lr3e-4_formatted_redo_shuffle_eval_lorar128_codeonlyloss_incontext/checkpoint-50/ # --checkpoint-path finetuned_models/merged_codellama7b_round1_lr3e-4_formatted_redo_shuffle_eval_lorar128_codeonlyloss_checkpoint-95 # _1example --examples-path round1_formatted_csv/examples.json --random-retrieval --apps