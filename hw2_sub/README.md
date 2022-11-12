## Download 
```
bash download.sh
```
To download the trained models with best performance, related data, and learning curves

## Preprocessing
```
python3 ./preprocess_model.py
```
To download pretrained models and tokenizers

## Training
### For multiple choice :
```
python3 ./train_multiple_choice.py
```
validation rate: 0.973, the model will be saved in the folder "ckpt/"

### For question answering :
```
python3 ./train_qa.py
```
validation rate: 0.804, the public score on Kaggle: 0.783, and the model will be saved in the folder "ckpt/"

## Testing
```
bash run.sh
```
For multiple choice: there will be a "relevant.json" in the folder "ckpt/" after testing 

For question answering: there will be a "predict.csv" in the current path after testing