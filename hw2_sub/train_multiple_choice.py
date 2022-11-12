from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torch import optim

from datasets import MCDataset, NSampler
from utils import fix_rand_seed, use_gpu, worker_init_fn, mask_tokens
import numpy as np

from transformers import BertForMultipleChoice
import logging
from math import ceil

logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

TRAIN = "train"
VALID = "valid"
SPLITS = [TRAIN, VALID]

REAL_BATCH_SIZE = 1


def train(model, optimizer, loader):
    model.train()
    train_acc = 0
    train_loss = 0
    batch_count = 0
    batch_idx = 1
    pbar = tqdm(loader)
    for batch_data in pbar:
        data_dict = {key: value.to(device) for key, value in batch_data['data'].items()}
        labels = batch_data['label'].to(device)

        outputs = model(**data_dict, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        loss.backward()
        batch_count += REAL_BATCH_SIZE

        preds = logits.argmax(dim=1)
        train_acc += preds.eq(labels).sum().item()
        train_loss += loss.item()
        if batch_count == args.batch_size:
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f'Batch [{batch_idx}/{ceil(len(loader.dataset) / args.batch_size)}]')
            pbar.set_postfix(loss=train_loss * REAL_BATCH_SIZE / (args.batch_size * (batch_idx + 1)),
                             acc=train_acc / (args.batch_size * (batch_idx + 1)))
            batch_idx += 1
            batch_count = 0
    return train_acc / len(loader.dataset), train_loss / len(loader)


def validation(model, loader):
    with torch.no_grad():
        model.eval()
        val_acc = 0
        val_loss = 0

        pbar = tqdm(enumerate(loader))
        for idx, batch_data in pbar:
            batch_input_ids = batch_data['data']['input_ids'].to(device)
            batch_token_type_ids = batch_data['data']['token_type_ids'].to(device)
            batch_attention_mask = batch_data['data']['attention_mask'].to(device)
            labels = batch_data['label'].to(device)
            logits = None
            for batch_idx in range(0, len(batch_data['data']['input_ids']), REAL_BATCH_SIZE):
                output = model(input_ids=batch_input_ids[batch_idx: batch_idx + REAL_BATCH_SIZE],
                               token_type_ids=batch_token_type_ids[batch_idx: batch_idx + REAL_BATCH_SIZE],
                               attention_mask=batch_attention_mask[batch_idx: batch_idx + REAL_BATCH_SIZE],
                               labels=labels[batch_idx: batch_idx + REAL_BATCH_SIZE])
                if logits is None:
                    logits = output.logits
                else:
                    logits = torch.cat((logits, output.logits), dim=0)
                    val_loss += output.loss.item() / args.num_replicas
            logits = logits.sum(dim=0)
            preds = logits.argmax()
            val_acc += preds.eq(labels[0]).sum().item()

            pbar.set_description(f'Batch [{idx}/{len(loader.dataset)}]')
            pbar.set_postfix(loss=val_loss / (idx + 1), acc=val_acc / (idx + 1))
        return val_acc / len(loader.dataset), val_loss / len(loader)


def main():
    fix_rand_seed()

    # files path
    context_path = args.data_dir / 'context.json'
    ckpt_path = args.ckpt_dir / args.model_name
    output_path = args.ckpt_dir / 'mengzi_mask'
    data_paths = {
        split: args.data_dir / f'{split}.json'
        for split in SPLITS
    }

    # dataset , dataloader
    train_datasets = MCDataset(data_path=data_paths[TRAIN], context_path=context_path,
                                  tokenizer_dir=ckpt_path, max_len=args.max_len, is_train=True)
    # print(train_datasets[0])

    val_datasets = MCDataset(data_path=data_paths[VALID], context_path=context_path,
                                tokenizer_dir=ckpt_path, max_len=args.max_len, is_train=False)
    val_sampler = NSampler(data_size=len(val_datasets), num_replicas=args.num_replicas)
    loaders: Dict[str, DataLoader] = {
        TRAIN: DataLoader(dataset=train_datasets, batch_size=REAL_BATCH_SIZE, num_workers=0, pin_memory=False,
                          worker_init_fn=worker_init_fn, shuffle=True, collate_fn=train_datasets.collate_fn),
        VALID: DataLoader(dataset=val_datasets, batch_size=args.num_replicas, sampler=val_sampler,
                          num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
                          collate_fn=val_datasets.collate_fn)
    }

    # model 
    modelMC = BertForMultipleChoice.from_pretrained(ckpt_path).to(device)

    # optimizer
    optimizer = optim.AdamW(params=modelMC.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, eps=1e-4)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_val_acc = 0
    for epoch in epoch_pbar:
        # train
        train_acc, train_loss = train(modelMC, optimizer=optimizer, loader=loaders[TRAIN])
        logging.info(f'epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}')
        logging.info(f'training_loss: {train_loss:.4f}, training_accuracy: {train_acc:.3f}')

        # validation
        val_acc, val_loss = validation(modelMC, loader=loaders[VALID])
        logging.info(f'validation_loss: {val_loss:.4f}, validation_accuracy: {val_acc:.3f}')
        if val_acc > best_val_acc:
            # update the best
            logging.info(f'Validation accuracy rate has improved from the last best performance {best_val_acc:.3f}')
            best_val_acc = val_acc
            # threshold : 90%
            if best_val_acc > 0.9:
                model_name = f'{str(output_path)}_{epoch + 1}_accuracy_{val_acc:.3f}'
                logging.info(f'The model {model_name} has been saved into ckpt')
                modelMC.save_pretrained(model_name)
        else:
            logging.info('No! The validation accuracy still not raise')

        scheduler.step(val_loss)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the context dataset.",
        default="./data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to the model file.",
        default="./ckpt",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of bert model.",
        default="mengzi-bert-base",
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)

    # model
    parser.add_argument("--weight_decay", type=float, default=5e-3)

    # optimizer
    parser.add_argument("--lr", type=float, default=4e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_replicas", type=int, default=3)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=10)

    _args = parser.parse_args()
    assert _args.batch_size % REAL_BATCH_SIZE == 0
    return _args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = use_gpu(args.device)
    main()

# python3 ./train_multiple_choice.py --num_epoch 10

