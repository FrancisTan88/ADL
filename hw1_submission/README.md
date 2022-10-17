# Environment Configuration
```
pip3 install -r requirements.in
```
To install the modules used later

# (Preprocessing Dataset)
```
bash preprocess.sh
```
To preprocess intent detectiona and slot tagging datasets, and a folder named "cache" will show up after executing the script, including embedding.pt, json file, and vocab.pkl for each assignment. Other than that, the txt file we download through "wget" will also show up.

# Download
```
bash download.sh
```
To download data, embeddings, and best weights that are used for training or testing models for each assignment

# Training
## For intent classification:
```
python3 train_intent.py --bidirectional --hidden_size 128 --drop_rate 0.4 --rnn_method GRU --max_len 35 --num_epoch 20
```
validation rate = 0.939, score on Kaggle = 0.92444
the results of model will be stored in the folder "ckpt/intent"

## For slot tagging:
```
python3 train_slot.py --max_len 40 --hidden_size 512 --drop_rate 0.4 --bidirectional --rnn_method LSTM --batch_size 64 --num_epoch 40
```
validation rate = 0.814, score on Kaggle = 0.79356
the results of model will be stored in the folder "ckpt/slot"

# Evaluation Report
```
python3 ./seq_eval.py --num_layers 2 --test_file data/slot/eval.json --ckpt_path ckpt/slot/bestPerformanceModel.pth --rnn_method LSTM  --data_type slot --hidden_size 512
```
![](https://i.imgur.com/yKc2vnZ.png)
To evaluate the slot tagging model(Q3) on validation set


