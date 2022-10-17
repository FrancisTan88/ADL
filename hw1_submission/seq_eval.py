import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import seqeval.scheme
import torch

from dataset import SeqTaggingClsDataset, SeqClsDataset
from utils import Vocab
from torch.utils.data import DataLoader
import numpy as np
from model import SeqTagger
import csv

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


SLOT = 'slot'

def test_model(model, loader, data_type, index2label):
    model.eval()
    predicts = []
    to_label = []
    for batch_data in loader:
        data_embedding = batch_data['data'].to(args.device)
        output = model(data_embedding, data_type)
        preds = output.argmax(dim=1)
        labels = batch_data['label']
        for pred, origin_len, label in zip(preds, batch_data['len'], labels):
            predicts.append(list(map(lambda x: index2label[x], pred.cpu().detach().numpy().tolist()[:origin_len])))
            to_label.append(list(map(lambda x: index2label[x], label.numpy().tolist()[:origin_len])))
    return to_label, predicts


def main(args):
    with open(args.cache_dir / args.data_type / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    label_idx_path = args.cache_dir / args.data_type / 'tag2idx.json'
    label2idx: Dict[str, int] = json.loads(label_idx_path.read_text())
    idx2label = {value: key for key, value in label2idx.items()}

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, label2idx, args.slot_max_len)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, pin_memory=False,
                        shuffle=False, collate_fn=dataset.collate_fn)
    
    path = args.cache_dir / SLOT / 'tag2idx.json'
    label2idx_len = len(json.loads(path.read_text()))
    embeddings = torch.load(args.cache_dir / SLOT / "embeddings.pt")

    model_tagger = SeqTagger(input_size=300, hidden_size=args.hidden_size, output_size=label2idx_len,
                            num_layer=args.num_layers, bidirectional=args.bidirectional, drop_rate=args.drop_rate,
                            rnn_method=args.rnn_method, embeddings=embeddings, pad_id=vocab.pad_id, device=args.device).to(args.device)
    model_tagger.load_state_dict(torch.load(args.ckpt_path))

    to_label, predicts = test_model(model_tagger, loader, args.data_type, idx2label)
    joint_acc = 0
    token_acc = 0
    element_count = 0
    for label, predict in zip(to_label, predicts):
        joint_acc += label == predict
        for e_l, e_p in zip(label, predict):
            element_count += 1
            token_acc += e_l == e_p
    print(f'Joint Accuracy = {joint_acc} / {len(to_label)} ({joint_acc/len(to_label)})')
    print(f'Token Accuracy = {token_acc} / {element_count} ({token_acc/element_count})')
    print(classification_report(to_label, predicts, scheme=seqeval.scheme.IOB2, mode='strict'))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--pred_file", type=Path, default="pred.csv")

    parser.add_argument("--slot_max_len", type=int, default=40)

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--drop_rate", type=float, default=0.4)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--rnn_method", type=str, default='LSTM')

    parser.add_argument("--f", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    args.pred_file = Path(f'pred.{args.data_type}.csv')
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
