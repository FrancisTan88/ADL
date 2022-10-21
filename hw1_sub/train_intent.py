import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import logging

import torch
from tqdm import trange
from torch.utils.data import DataLoader
from torch import optim

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def train_model(model, optimizer, criterion, dataloader):
    model.train()
    total_match = 0
    total_loss = 0
    for batch in dataloader:
        index = batch['data'].to(args.device)
        label = batch['label'].to(args.device)
        optimizer.zero_grad()
        result = model(index)

        # calculate loss (CrossEntropy)
        loss = criterion(result, label)
        loss.backward()
        optimizer.step()
        
        # get idx of max element
        max_prediction = result.argmax(dim=1)
        total_match += max_prediction.eq(label).sum().item()
        total_loss += loss.item()
    
    accuracy_rate = total_match/len(dataloader.dataset)
    loss_rate = total_loss/len(dataloader)
    return accuracy_rate, loss_rate


def validation(model, criterion, dataloader):
    model.eval()
    total_match = 0
    total_loss = 0
    for batch in dataloader:
        index = batch['data'].to(args.device)
        label = batch['label'].to(args.device)
        result = model(index)
        loss = criterion(result, label)

        max_prediction = result.argmax(dim=1)
        total_match += max_prediction.eq(label).sum().item()
        total_loss += loss.item()

    accuracy_rate = total_match/len(dataloader.dataset)
    loss_rate = total_loss/len(dataloader)
    return accuracy_rate, loss_rate


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    # load "intent2idx.json"
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text()) 
    # load json in "data"
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}  # {train: train.json, eval: eval.json}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    # load embeddings
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }  # {train: SeqClsDataset(train.json), eval: SeqClsDataset(eval.json)}

    # TODO: crecate DataLoader for train / dev datasets  
    dataloader: Dict[str, DataLoader] = {}
    for name, dataset in datasets.items():
        dataloader[name] = DataLoader(dataset=dataset, batch_size=args.batch_size, pin_memory=False,
                                shuffle=True, collate_fn=dataset.collate_fn)

    # TODO: init model and move it to target device(cpu / gpu)
    model_intent = SeqClassifier(input_size=300, hidden_size=args.hidden_size, output_size=datasets[TRAIN].num_classes,
                            num_layer=args.num_layers, bidirectional=args.bidirectional, drop_rate=args.drop_rate,
                            rnn_method=args.rnn_method, embeddings=embeddings, pad_id=vocab.pad_id, device=args.device).to(args.device)

    # TODO: init optimizer
    optimizer = optim.AdamW(params=model_intent.parameters(), lr=args.lr)
    # loss func
    criterion = torch.nn.CrossEntropyLoss()

    bestAccuracy = 0
    epoch_show = trange(args.num_epoch, desc="Epoch")
    for e in epoch_show:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        trainingAccuracy, trainingLoss = train_model(model_intent, optimizer=optimizer, criterion=criterion, dataloader=dataloader[TRAIN])
        logging.info(f'training loss rate: {trainingLoss:.4f}, training accuracy rate: {trainingAccuracy:.3f}')

        validationAccuracy, validationLoss = validation(model_intent, criterion=criterion, dataloader=dataloader[DEV])
        logging.info(f'validation loss rate: {validationLoss:.4f}, validation accuracy rate: {validationAccuracy:.3f}')
 
        if validationAccuracy > bestAccuracy:
            logging.info(f'validation accuracy rate is increasing, and the last best performance was {bestAccuracy:.3f}')
            bestAccuracy = validationAccuracy  # update the best rate
            if bestAccuracy > 0.88:
                model_save = 'bestPerformanceModel.pth'
                logging.info(f'{model_save} has been saved')
                torch.save(model_intent.state_dict(), args.ckpt_dir / model_save)
        else:
            logging.info('validation accuracy rate is unchanged')


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--bidirectional", action='store_true', default=False)
    parser.add_argument("--rnn_method", type=str, default='RNN')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
