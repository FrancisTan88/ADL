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
from math import ceil
from transformers import AutoModelForQuestionAnswering, AutoConfig, BertForQuestionAnswering
from matplotlib import pyplot as plt
from collections import defaultdict


logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

TRAIN = "train"
VALID = "valid"
SPLITS = [TRAIN, VALID]

REAL_BATCH_SIZE = 2

def loss_calculation(start_end_disparity, loss_type='no'):
    start_end_disparity = start_end_disparity.clamp(min=0)
    return torch.ones(start_end_disparity.shape)

def plt_curve(epoch, rec):
    fig = plt.figure(figsize=(9, 4))
    fig.suptitle(f"{args.model_name} {args.num_epoch}epoch")

    # loss curve
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Loss')
    ax.plot(rec['train_loss'], label='train')
    ax.plot(rec['val_loss'], label='validate')
    ax.set_xlim(1, epoch)
    ax.legend()

    # exact match curve
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('EM')
    ax.plot(rec['train_acc'], label='train')
    ax.plot(rec['val_acc'], label='validate')
    ax.set_xlim(1, epoch)
    ax.legend()

    fig.savefig(f'pretrained_curve_{epoch}.png')
    plt.close(fig)

def train(model, optimizer, loader):
    model.train()
    train_acc = 0
    train_loss = 0
    batch_count = 0
    batch_idx = 1
    data_amount = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    pbar = tqdm(loader)
    for batch_data in pbar:
        data_dict = {key: value.to(device) for key, value in batch_data['data'].items()}
        labels = {key: value.to(device) for key, value in batch_data['labels'].items()}
        outputs = model(**data_dict)
        start_positions = outputs.start_logits.argmax(dim=1)
        end_positions = outputs.end_logits.argmax(dim=1)
        loss_weights = loss_calculation(start_positions - end_positions, loss_type=args.loss_type)
        loss = 0
        for start_logit, end_logit, start_label, end_label, loss_weight in zip(outputs.start_logits, outputs.end_logits,
                                                                               labels['start_positions'],
                                                                               labels['end_positions'],
                                                                               loss_weights):
            start_loss = loss_fn(start_logit.unsqueeze(0), start_label.unsqueeze(0))
            end_loss = loss_fn(end_logit.unsqueeze(0), end_label.unsqueeze(0))
            loss += (start_loss + end_loss) * loss_weight / len(loss_weights)
        
        loss.backward()
        batch_count += REAL_BATCH_SIZE
        batch_data['cls_indices'] = batch_data['cls_indices'].to(device)
        real_indices = torch.logical_not(
            torch.logical_and(batch_data['cls_indices'].eq(labels['start_positions']),batch_data['cls_indices'].eq(labels['end_positions']))
        )
        accuracy = torch.logical_and(start_positions.eq(labels['start_positions']),
                                end_positions.eq(labels['end_positions']))
        accuracy = accuracy[real_indices]
        data_amount += len(accuracy)
        train_acc += accuracy.sum().item()
        train_loss += loss.item()
        if batch_count == args.batch_size:
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f'Batch [{batch_idx}/{ceil(len(loader.dataset) / args.batch_size)}]')
            pbar.set_postfix(loss=train_loss * REAL_BATCH_SIZE / (args.batch_size * (batch_idx + 1)),
                             acc=train_acc / data_amount)
            batch_idx += 1
            batch_count = 0
    return train_acc/data_amount, train_loss/len(loader)


def validation(model, loader):
    with torch.no_grad():
        model.eval()
        val_acc = 0
        val_loss = 0
        real_data_count = 0
        pbar = tqdm(enumerate(loader))
        for idx, batch_data in pbar:
            data_dict = {key: value.to(device) for key, value in batch_data['data'].items()}
            labels = {key: value.to(device) for key, value in batch_data['labels'].items()}
            outputs = model(**data_dict, **labels)
            loss = outputs.loss

            start_positions = outputs.start_logits.argmax(dim=1)
            end_positions = outputs.end_logits.argmax(dim=1)
            batch_data['cls_indices'] = batch_data['cls_indices'].to(device)
            real_indices = torch.logical_not(
                torch.logical_and(batch_data['cls_indices'].eq(labels['start_positions']),
                                  batch_data['cls_indices'].eq(labels['end_positions']))
            )
            acc = torch.logical_and(start_positions.eq(labels['start_positions']),
                                    end_positions.eq(labels['end_positions']))
            acc = acc[real_indices]
            real_data_count += len(acc)
            val_acc += acc.sum().item()
            val_loss += loss.item()
            pbar.set_description(f'Batch [{idx}/{len(loader.dataset)}]')
            pbar.set_postfix(loss=val_loss / (idx + 1), acc=val_acc / real_data_count)
        return val_acc / real_data_count, val_loss / len(loader)


def main():
    fix_rand_seed()

    ckpt_path = args.ckpt_dir / args.model_name
    context_path = args.data_dir / 'context.json'
    output_path = args.ckpt_dir / args.model_name
    data_paths = {}
    for s in SPLITS:
        data_paths[s] = args.data_dir / f'{s}.json'

    datasets: Dict[str, QADataset] = {
        s: QADataset(data_path=path, context_path=context_path,
                         tokenizer_dir=ckpt_path, max_len=args.max_len, doc_stride=args.doc_stride)
        for s, path in data_paths.items()
    }

    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(dataset=split_dataset, batch_size=REAL_BATCH_SIZE, num_workers=0, pin_memory=False,
                          worker_init_fn=worker_init_fn, shuffle=False, collate_fn=split_dataset.collate_fn)
        for split, split_dataset in datasets.items()
    }

    # for the model without pretraining
    # modelConfig = AutoConfig.from_pretrained(ckpt_path)
    # modelQA = BertForQuestionAnswering(config=modelConfig).to(device)

    # model
    modelQA = AutoModelForQuestionAnswering.from_pretrained(ckpt_path).to(device)
    # optimizer
    optimizer = optim.AdamW(params=modelQA.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, eps=1e-6)

    progress_bar = trange(args.num_epoch, desc="Epoch")
    best_val_acc = 0
    record = defaultdict(list)
    for epoch in progress_bar:
        logging.info(f'epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}')
        train_acc, train_loss = train(modelQA, optimizer=optimizer, loader=dataloaders[TRAIN])
        logging.info(f'training_loss: {train_loss:.4f}, training_accuracy: {train_acc:.3f}')

        val_acc, val_loss = validation(modelQA, loader=dataloaders[VALID])
        logging.info(f'validation_loss: {val_loss:.4f}, validation_accuracy: {val_acc:.3f}')

        # update the best rate
        if val_acc > best_val_acc:
            logging.info(f'Validation accuracy rate has improved from the last best performance {best_val_acc:.3f}')
            best_val_acc = val_acc
            # threshold = 75%
            if best_val_acc > 0.75:
                model_name = f'{str(output_path)}_{epoch + 1}_accuracy_{val_acc:.3f}'
                logging.info(f'The model {model_name} has been saved into ckpt')
                modelQA.save_pretrained(model_name)
        else:
            logging.info('No! The validation accuracy still not raise')

        scheduler.step(val_loss)

        # plot the curve
        record['train_acc'].append(train_acc)
        record['val_acc'].append(val_acc)
        record['train_loss'].append(train_loss)
        record['val_loss'].append(val_loss)
        plt_curve(epoch + 1, record)


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
        default="chinese-macbert-base",
    )

    # data
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--doc_stride", type=int, default=128)

    # optimizer
    parser.add_argument("--lr", type=float, default=4e-5)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--loss_type", type=str, default='no')

    _args = parser.parse_args()
    assert _args.batch_size % REAL_BATCH_SIZE == 0
    return _args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    device = use_gpu(args.device)
    main()
