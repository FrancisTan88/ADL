#!/bin/bash
# {1}: path to validation jsonl file
# {2}: output path of prediction file 
python3 run_summarization_no_trainer.py --model_name_or_path ckpt --text_column maintext --validation_file ${1} --output_path ${2} --num_beams 8

# python3 run_summarization_no_trainer.py --model_name_or_path ckpt --text_column maintext --validation_file data/public.jsonl --output_path sum_beam_search.jsonl --num_beams 8