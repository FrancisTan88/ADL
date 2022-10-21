import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch.utils.data import DataLoader
import csv
import numpy as np

def test_model(model, dataloader):
    model.eval()
    predicts_res = []
    idxes = []
    for batch in dataloader:
        data_intent_index = batch['data'].to(args.device)
        result = model(data_intent_index)
        max_prediction = result.argmax(dim=1)
        predicts_res += max_prediction.cpu().detach().numpy().tolist()
        idxes += batch['idx']

    return idxes, predicts_res


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())
    dict_idxTointent = {value: key for key, value in intent2idx.items()}

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO: crecate DataLoader for test dataset
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, pin_memory=False,
                         shuffle=False, collate_fn=dataset.collate_fn)

    # model
    model_intent = SeqClassifier(input_size=300, hidden_size=args.hidden_size, output_size=dataset.num_classes,
                            num_layer=args.num_layers, bidirectional=args.bidirectional, drop_rate=args.drop_rate,
                            rnn_method=args.rnn_method, embeddings=embeddings, pad_id=vocab.pad_id, device=args.device).to(args.device)
    model_intent.load_state_dict(torch.load(args.ckpt_path))

    # predict
    idxes, predicts_res = test_model(model_intent, dataloader)
    
    # write prediction to file (args.pred_file)
    with open(args.pred_file, "w", encoding="utf-8", newline='') as new_file:
        writer = csv.writer(new_file)
        # header
        writer.writerow(['id', 'intent'])
        # content
        for test_id, pred in zip(idxes, predicts_res):
            intent = dict_idxTointent[pred]
            writer.writerow([test_id, intent])


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
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.csv")

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
