from abc import ABC
from typing import List, Dict

import torch
from torch.utils.data import Dataset, Sampler

from transformers import BertTokenizerFast

from pathlib import Path
import json
import random
from collections import defaultdict
import copy
from functools import partial
from utils import mask_tokens


class BaseDataset(Dataset, ABC):
    def __init__(
            self,
            data_path: Path,
            context_path: Path,
            tokenizer_dir: str,
            max_len: int,
            is_train: bool = True
    ):
        self.data: List[Dict] = json.loads(data_path.read_text(encoding='utf-8'))
        self.context: List[str] = json.loads(context_path.read_text(encoding='utf-8'))
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_dir)
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.data)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        raise NotImplementedError('Please implement collate function')


class MCDataset(BaseDataset):
    def __init__(
            self,
            data_path: Path,
            context_path: Path,
            tokenizer_dir: str,
            max_len: int,
            is_train: bool = True
    ):
        super(MCDataset, self).__init__(data_path, context_path, tokenizer_dir, max_len, is_train)
        for ins in self.data:
            ins['label_encoder']: Dict = dict()
            ins['label_decoder']: Dict = dict()
            for label, para_idx in enumerate(ins['paragraphs']):  # e.g. (0, 2018), (1, 6952)
                # add encoder and decoder in instance
                ins['label_encoder'][para_idx] = label  
                ins['label_decoder'][label] = para_idx 
                # find the possible paragraph from context and throw it back
                paragraph = self.context[para_idx]  
                ins['paragraphs'][label] = paragraph  
    
    def __getitem__(self, index):
        instance = copy.deepcopy(self.data[index])
        instance['paragraphs'] = list(map(partial(self.truncation, question_len=len(instance['question'])),
                                          instance['paragraphs']))
        return instance

    # get random paragraph
    def truncation(self, paragraph, question_len):

        def crop(para, start, end):
            if start_index < end_index:
                para = para[start: end]
            else:
                para = para[: end] + para[start:]
            return para

        max_len = self.max_len - question_len
        max_len -= 3
        start_index = random.randint(0, len(paragraph) - 1)
        # the para length over the max len contraint
        if len(paragraph) > max_len:
            end_index = (start_index + max_len) % len(paragraph)
            paragraph = crop(paragraph, start_index, end_index)

        return paragraph

    
    def collate_fn(self, samples: List[Dict]) -> Dict:
        batch_encode_data = defaultdict(list)
        batch_labels = []
        batch_ids = []
        batch_label_decoders = []
        for instance in samples:
            # use tokenizer
            encode_data = self.tokenizer([instance['question']] * len(instance['paragraphs']), instance['paragraphs'],
                                         max_length=self.max_len, truncation='only_second', padding='max_length',
                                         return_tensors='pt', return_special_tokens_mask=True)
            special_tokens_mask = encode_data.pop('special_tokens_mask')
            batch_labels.append(instance['label_encoder'].get(instance.get('relevant', -1), -1)) # get label of relevant artical 
            # for training data
            if self.is_train:
                encode_data['input_ids'] = mask_tokens(encode_data['input_ids'],
                                                       paragraph_indices=(encode_data['token_type_ids'] &
                                                                          ~special_tokens_mask).bool(),
                                                       mask_id=self.tokenizer.mask_token_id, mask_prob=0.15)
            else:
                batch_ids.append(instance['id'])
                batch_label_decoders.append(instance['label_decoder'])
            for k, v in encode_data.items():
                batch_encode_data[k].append(v)

        batch_encode_data = {k: torch.stack(v) for k, v in batch_encode_data.items()}
        batch_labels = torch.LongTensor(batch_labels)  # to tensor
        data = {}
        data['data'] = batch_encode_data
        data['label'] = batch_labels
        if self.is_train is False:
            data['ids'] = batch_ids
            data['label_decoders'] = batch_label_decoders
        return data


def transform_index(start_idx, end_idx, sequence_ids, offset_mapping, cls_index):
    # Find the start and end position
    token_start_idx = 0
    while sequence_ids[token_start_idx] != 1:
        token_start_idx += 1

    token_end_idx = token_start_idx
    while sequence_ids[token_end_idx] is not None:
        token_end_idx += 1
    token_end_idx -= 1

    if offset_mapping[token_start_idx][0] <= start_idx and offset_mapping[token_end_idx][1] >= end_idx:
        while token_start_idx < len(offset_mapping) and offset_mapping[token_start_idx][0] <= start_idx:
            token_start_idx += 1
        start_position = token_start_idx - 1

        while token_end_idx >= 0 and offset_mapping[token_end_idx][1] >= end_idx:
            token_end_idx -= 1
        end_position = token_end_idx + 1

        return start_position, end_position

    else:
        return cls_index, cls_index


class QADataset(BaseDataset):
    def __init__(
            self,
            data_path: Path,
            context_path: Path,
            tokenizer_dir: str,
            max_len: int,
            doc_stride,
            relevant_path: Path = None,
            is_train: bool = True
    ):
        super(QADataset, self).__init__(data_path, context_path, tokenizer_dir, max_len, is_train)
        self.doc_stride = doc_stride

        if relevant_path is not None:
            paragraph_indices: Dict[str, int] = json.loads(relevant_path.read_text(encoding='utf-8'))

        for ins in self.data:
            if relevant_path is None:
                para_idx = ins['relevant']
            else:
                para_idx = paragraph_indices[ins['id']]
            ins['paragraphs'] = self.context[para_idx]

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, samples: List[Dict]) -> Dict:
        data_size = len(samples)
        data = {}
        data['input_ids'] = []
        data['token_type_ids'] = []
        data['attention_mask'] = []
        # data = {
        #     key: [] for key in ['input_ids', 'token_type_ids', 'attention_mask']
        # }
        labels = defaultdict(list)
        ids = [None] * data_size
        paragraghs = [None] * data_size
        num_slices = [None] * data_size
        offsets_mappings = []
        cls_indices = []
        special_tokens_masks = []
        for idx, instance in enumerate(samples):
            input_encoding = self.tokenizer(instance['question'], instance['paragraphs'], max_length=self.max_len,
                                            truncation='only_second', padding='max_length', return_tensors='pt',
                                            add_special_tokens=True, return_offsets_mapping=True,
                                            stride=self.doc_stride, return_overflowing_tokens=True,
                                            return_special_tokens_mask=True)
            offsets_mapping = input_encoding.pop('offset_mapping')
            num_slice = len(input_encoding.pop('overflow_to_sample_mapping'))
            num_slices[idx] = num_slice
            special_tokens_mask = input_encoding.pop('special_tokens_mask')
            # if training
            if self.is_train is True:
                # find start and end position of instance's answer
                start_idx = instance['answer']['start']
                end_idx = instance['answer']['start'] + len(instance['answer']['text'])
                for slice_idx in range(num_slice):
                    for k in data.keys():
                        data[k].append(input_encoding[k][slice_idx])
                    offsets = offsets_mapping[slice_idx]
                    cls_idx = input_encoding['input_ids'][slice_idx].tolist().index(self.tokenizer.cls_token_id)
                    cls_indices.append(cls_idx)
                    start_position, end_position = transform_index(start_idx, end_idx,
                                                                   input_encoding.sequence_ids(slice_idx),
                                                                   offsets, cls_idx)
                    labels['start_positions'].append(start_position)
                    labels['end_positions'].append(end_position)
            else:
                ids[idx] = instance['id']
                paragraghs[idx] = instance['paragraphs']
                for slice_idx in range(num_slice):
                    for k in data.keys():
                        data[k].append(input_encoding[k][slice_idx])
                    special_tokens_masks.append(special_tokens_mask[slice_idx])
                    offsets_mappings.append(offsets_mapping[slice_idx])

        data = {k: torch.stack(v) for k, v in data.items()}
        batch_data = {
            'data': data,
            'num_slices': num_slices
        }
        if self.is_train:
            labels = {k: torch.LongTensor(v) for k, v in labels.items()}
            batch_data['labels'] = labels
            batch_data['cls_indices'] = torch.tensor(cls_indices)
        else:
            batch_data['ids'] = ids
            batch_data['paragraphs'] = paragraghs
            batch_data['offset_mapping'] = torch.stack(offsets_mappings)
            batch_data['special_token_masks'] = torch.stack(special_tokens_masks)
        return batch_data


class NSampler(Sampler):
    def __init__(self, data_size, num_replicas):
        self.data_size = data_size
        self.num_replicas = num_replicas

    def __iter__(self):
        return iter(map(lambda x: int(x / self.num_replicas), range(self.data_size * self.num_replicas)))

    def __len__(self):
        return self.data_size * self.num_replicas
