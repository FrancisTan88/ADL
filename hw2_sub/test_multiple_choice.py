import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader

from datasets import MCDataset, NSampler
from utils import fix_rand_seed, use_gpu, worker_init_fn, mask_tokens
import numpy as np

from transformers import BertForMultipleChoice


REAL_BATCH_SIZE = 16


def test(model, loader):
    with torch.no_grad():
        model.eval()
        data = dict()
        pbar = tqdm(loader)
        for batch_data in pbar:
            batch_input_ids = batch_data['data']['input_ids'].to(device)
            batch_token_type_ids = batch_data['data']['token_type_ids'].to(device)
            batch_attention_mask = batch_data['data']['attention_mask'].to(device)
            logits = None
            for batch_idx in range(0, len(batch_data['data']['input_ids']), REAL_BATCH_SIZE):
                output = model(input_ids=batch_input_ids[batch_idx: batch_idx + REAL_BATCH_SIZE],
                               token_type_ids=batch_token_type_ids[batch_idx: batch_idx + REAL_BATCH_SIZE],
                               attention_mask=batch_attention_mask[batch_idx: batch_idx + REAL_BATCH_SIZE])
                if logits is None:
                    logits = output.logits
                else:
                    logits = torch.cat((logits, output.logits), dim=0)
            logits = logits.sum(dim=0)
            pred_label = logits.argmax().item()
            data.update({batch_data['ids'][0]: batch_data['label_decoders'][0][pred_label]})
        return data


def main():
    fix_rand_seed()

    test_dataset = MCDataset(data_path=args.test_path, context_path=args.context_path,
                                tokenizer_dir=args.model_dir, max_len=args.max_len, is_train=False)
    test_sampler = NSampler(data_size=len(test_dataset), num_replicas=args.num_replicas)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.num_replicas, sampler=test_sampler,
                             num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
                             collate_fn=test_dataset.collate_fn)
    modelRoBETa = BertForMultipleChoice.from_pretrained(args.model_dir).to(device)

    data_preds = test(modelRoBETa, test_loader)
    json.dump(data_preds, args.relevant_path.open('w'), indent=2)


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
        help="Directory to the relevant json file.",
        default="./ckpt/relevant.json",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory to the model.",
        required=True
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # model
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    # data loader
    parser.add_argument("--num_replicas", type=int, default=5)

    # testing
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )

    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = parse_args()
    device = use_gpu(args.device)
    main()
