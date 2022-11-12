import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torch import optim

from datasets import QADataset
from utils import fix_rand_seed, use_gpu, worker_init_fn, mask_tokens
import numpy as np
import logging

from transformers import AutoModelForQuestionAnswering
import csv

REAL_BATCH_SIZE = 8
count = 0


def get_best_index(start_values, start_indices, end_values, end_indices, topk):
    num_slices = len(start_values)
    start_index = -1
    end_index = -1
    max_slice_index = -1
    max_logit_value = float('-inf')
    for slice_idx in range(num_slices):
        for start_idx in range(topk):
            for end_idx in range(topk):
                if start_indices[slice_idx, start_idx] > end_indices[slice_idx, end_idx]:
                    continue
                logit_value = start_values[slice_idx, start_idx] + end_values[slice_idx, end_idx]
                if logit_value > max_logit_value:
                    max_logit_value = logit_value
                    max_slice_index = slice_idx
                    start_index = start_indices[slice_idx, start_idx]
                    end_index = end_indices[slice_idx, end_idx]
    assert start_index != -1 and end_index != -1 and max_slice_index != -1
    return start_index, end_index, max_slice_index


def get_predictions(batch_data, start_logits, end_logits) -> list:
    predictions = []
    # change special tokens' prediction value to -inf
    special_token_masks = batch_data['special_token_masks'].type(torch.bool)
    start_logits[special_token_masks] = -torch.inf
    end_logits[special_token_masks] = -torch.inf

    end_slice_idx = 0
    for idx, num_slice in enumerate(batch_data['num_slices']):
        start_slice_idx = end_slice_idx
        end_slice_idx += num_slice
        paragraph = batch_data['paragraphs'][idx]
        offset_mapping = batch_data['offset_mapping'][start_slice_idx: end_slice_idx]
        # get max prob predict of each slice in one paragraph
        topk = 3
        start_values, start_indices = start_logits[start_slice_idx: end_slice_idx].topk(dim=1, k=topk)
        end_values, end_indices = end_logits[start_slice_idx: end_slice_idx].topk(dim=1, k=topk)
        start_index, end_index, slice_index = get_best_index(start_values, start_indices, end_values, end_indices, topk)
        prediction = paragraph[offset_mapping[slice_index, start_index][0]: offset_mapping[slice_index, end_index][1]]
        predictions.append([batch_data['ids'][idx], prediction])
    return predictions


def test(model, loader):
    with torch.no_grad():
        model.eval()
        results = [['id', 'answer']]
        pbar = tqdm(loader)
        for batch_data in pbar:
            data_dict = {key: value.to(device) for key, value in batch_data['data'].items()}
            outputs = model(**data_dict)
            predictions = get_predictions(batch_data, outputs.start_logits, outputs.end_logits)
            results += predictions
        return results


def main():
    fix_rand_seed()

    global test_datasets
    test_datasets = QADataset(data_path=args.test_path, context_path=args.context_path, max_len=args.max_len,
                              relevant_path=args.relevant_path, tokenizer_dir=args.model_dir, is_train=False,
                              doc_stride=args.doc_stride)

    test_loaders = DataLoader(dataset=test_datasets, batch_size=REAL_BATCH_SIZE, num_workers=0, pin_memory=False,
                              worker_init_fn=worker_init_fn, shuffle=False, collate_fn=test_datasets.collate_fn)
    modelQA = AutoModelForQuestionAnswering.from_pretrained(args.model_dir).to(device)

    predictions = test(modelQA, test_loaders)
    global count
    print(count)
    with args.output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(predictions)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--context_path",
        type=Path,
        help="Path to the context dataset.",
        default="./data/context.json",
    )
    parser.add_argument(
        "--test_path",
        type=Path,
        help="Path to the test dataset.",
        default="./data/test.json",
    )
    parser.add_argument(
        "--relevant_path",
        type=Path,
        help="Path to the relevant json file.",
        default="./ckpt/relevant.json",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory to the model.",
        required=True
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to the output json file.",
        default="./predict.csv"
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--doc_stride", type=int, default=128)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=20)

    _args = parser.parse_args()
    assert _args.batch_size % REAL_BATCH_SIZE == 0
    return _args


if __name__ == "__main__":
    args = parse_args()
    device = use_gpu(args.device)
    main()
