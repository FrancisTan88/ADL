# ADL22-HW3
Dataset & evaluation script for ADL 2022 homework 3

## Dataset
[download link](https://drive.google.com/file/d/186ejZVADY16RBfVjzcMcz9bal9L3inXC/view?usp=sharing)

## Installation
```
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
pip install -r requirements.txt
```


## Usage
### train model
```
python run_summarization.py 
--model_name_or_path google/mt5-small 
--train_file ${train jsonl file}
--validation_file ${validation jsonl file}
--output_dir ${output dir}
--text_column maintext --summary_column title 
--per_device_train_batch_size=2 
--per_device_eval_batch_size=2 
--num_train_epochs 20 
--learning_rate 1e-4 
--gradient_accumulation_steps 8 
--num_beams 8
--do_train  --do_eval 
--predict_with_generate --overwrite_output_dir 
```

### get prediction jsonl file
```
python run_summarization_no_trainer.py 
--model_name_or_path ${model dir} 
--validation_file ${data path}
--output_path ${output path}
--text_column maintext 
```

