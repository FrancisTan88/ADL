#!/bin/bash
multiple_choice_path="ckpt/best_mengzi"
question_answering_path="ckpt/best_macbert"
relevant_path="ckpt/relevant.json"

python3 ./test_multiple_choice.py --context_path "${1}" --test_path "${2}" --relevant_path $relevant_path --model_dir $multiple_choice_path
python3 ./test_qa.py --context_path "${1}" --test_path "${2}" --output_path "${3}" --relevant_path $relevant_path --model_dir $question_answering_path 

# testing
# python3 ./test_multiple_choice.py --context_path data/context.json --test_path data/test.json --relevant_path ckpt/relevant_test2.json --model_dir $multiple_choice_path
# python3 ./test_qa.py --context_path data/context.json --test_path data/test.json --output_path predict.csv --relevant_path $relevant_path --model_dir $question_answering_path 