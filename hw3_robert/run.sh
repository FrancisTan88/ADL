#!/bin/bash
model_dir="ckpt"
python3.8 run_summarization_no_trainer.py --model_name_or_path $model_dir --text_column maintext --validation_file $1 --output_path $2 --num_beams 8