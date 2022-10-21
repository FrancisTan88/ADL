import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
import numpy as np

import csv


def test_model(model, dataloader):
    model.eval()
    predicts_res = []
    indexs = []
    for batch in dataloader:
        data = batch['data'].to(args.device)
        result = model(data)
        max_prediction = result.argmax(dim=1)
        
        tmp = []
        for pred, len in zip(max_prediction, batch['len']):
            tmp.append(pred.cpu().detach().numpy().tolist()[:len])
        predicts_res += tmp
        indexs += batch['idx']

    return indexs, predicts_res



def main(args):
    # TODO: implement main function
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())
    dict_idxTotag = {value: key for key, value in tag2idx.items()}

    data = json.loads(args.data_dir.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, pin_memory=False,
                         shuffle=False, collate_fn=dataset.collate_fn)
    model_tagger = SeqTagger(input_size=300, hidden_size=args.hidden_size, output_size=dataset.num_classes,
                            num_layer=args.num_layers, bidirectional=args.bidirectional, drop_rate=args.drop_rate,
                            rnn_method=args.rnn_method, embeddings=embeddings, pad_id=vocab.pad_id, device=args.device).to(args.device)
    model_tagger.load_state_dict(torch.load(args.ckpt_dir))

    idxes, predicts_res = test_model(model_tagger, dataloader)
    with args.pred_file.open("w", encoding="utf-8", newline='') as new_file:
        writer = csv.writer(new_file)
        writer.writerow(['id', 'tags'])
        for test_id, pred in zip(idxes, predicts_res):
            tag = ' '.join([dict_idxTotag[p] for p in pred])
            writer.writerow([test_id, tag])
            


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_method", type=str, default='RNN')

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)