# HW3

## Download
To download the data, model weights, and learning curve, etc
```
bash download.sh
```


## Install
For tw_rouge
```
git clone https://github.com/moooooser999/ADL22-HW3.git
cd ADL22-HW3
pip install -e tw_rouge
```
For some required modules
```
pip3 install -r requirements.txt
```


## Training
To train summarization model
```
python3 run_summarization.py 
--model_name_or_path google/mt5-small 
--train_file ${path to training data file}
--validation_file ${path to validation data file}
--output_dir ${output directory}
--text_column maintext 
--summary_column title 
--per_device_train_batch_size=2 
--per_device_eval_batch_size=2 
--num_train_epochs 20 
--learning_rate 1e-4 
--gradient_accumulation_steps 8 
--num_beams 8
--do_train 
--do_eval 
--predict_with_generate 
--overwrite_output_dir 
```

## Prediction
To predict titles of articles
```
bash run.sh
```
After running this command, there will be a jsonl file named "sum_beam_search", using that file as the prediction file to get the rouge score in the next step.

## Evaluation(Rouge Score)
To evaluate the prediction
```
python3 ./eval.py -r ${path to validation data file} -s ${path to prediction file}
```
