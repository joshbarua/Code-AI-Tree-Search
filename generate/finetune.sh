# python /home/sanjayss/joshbarua/code-hexagons/finetune.py --base-model codellama/CodeLlama-7b-Python-hf --lr 3e-4 --train-csv-path round1_formatted_probpath_redo_csv/train.csv --val-csv-path round1_formatted_unique_probpath_csv/val.csv --output-dir finetuned_models/codellama7b_round1_lr3e-4_formatted_redo_shuffle_eval_lorar128_codeonlyloss --lora_r 128 --shuffle --batch-size 8 --gradient-accumulation-steps 16 --eval-steps 5 --task apps --num-epochs 16 --loss-only-on-code --balanced-sampling # _1example --examples-path round1_formatted_csv/examples.json --random-retrieval --apps
# python /home/sanjayss/joshbarua/code-hexagons/finetune_with_incontext.py --base-model codellama/CodeLlama-7b-Python-hf --lr 3e-4 --train-csv-path round1_formatted_unique_probpath_csv/train.csv --val-csv-path round1_formatted_unique_probpath_csv/val.csv --output-dir finetuned_models/codellama7b_round1_lr3e-4_formatted_unique_shuffle_eval_lorar64_1example_greaterbetter --shuffle --lora_r 64 --task apps --batch-size 4 --gradient-accumulation-steps 32 --eval-steps 5 --task apps --num-epochs 16 --examples-path round1_formatted_csv/examples.json
# python /home/sanjayss/joshbarua/code-hexagons/finetune.py --base-model codellama/CodeLlama-7b-Python-hf --lr 3e-4 --train-csv-path round1_formatted_probpath_only_revision0.9_correction0.5_edit20_error_trace_csv/train.csv --val-csv-path round1_formatted_probpath_only_revision0.9_correction0.5_edit20_error_trace_csv/val.csv --output-dir finetuned_models/codellama7b_round1_lr3e-4_formatted_only_revision_0.9_0.5_shuffle_eval_lorar64_codeonlyloss --lora_r 64 --shuffle --batch-size 4 --gradient-accumulation-steps 32 --eval-steps 5 --task apps --num-epochs 60 --loss-only-on-code --balanced-sampling # _1example --examples-path round1_formatted_csv/examples.json --random-retrieval --apps
python /home/sanjayss/joshbarua/code-hexagons/finetune.py --base-model codellama/CodeLlama-7b-Python-hf --lr 3e-4 --train-csv-path round1_noawq_csv/train.csv --val-csv-path round1_noawq_csv/val.csv --output-dir finetuned_models/codellama7b_noawq_round1_lr3e-4_eval_lorar128 --lora_r 128 --shuffle --batch-size 8 --gradient-accumulation-steps 16 --eval-steps 5 --task apps --num-epochs 16 --loss-only-on-code --balanced-sampling # --in-context --resume # _1example --examples-path round1_formatted_csv/examples.json --random-retrieval --apps