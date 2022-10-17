from typing import List, Dict

from torch.utils.data import Dataset

import utils
from utils import Vocab

import torch


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping  
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}  # reverse
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    # convert training data to idxs
    def collate_fn(self, samples: List[Dict]) -> Dict:
        
        # TODO: implement collate_fn
        tokens = []
        labels = []
        indexs = []
        for instance in samples:
            tokens.append(instance['text'].split()) 
            labels.append(self.label2idx(instance.get('intent', None)))  # convert label to idx(i.e. 'intent2idx.json')
            indexs.append(instance['id'])
        data_after_pad = self.vocab.encode_batch(tokens, self.max_len) # convert words to index(and padding 0 if the length < max_len), data_after_pad: a 2D array
        data_after_pad = torch.LongTensor(data_after_pad) # to tensor
        labels = torch.LongTensor(labels)
        data = {
            'data': data_after_pad,   # 2D array of index(words)
            'label': labels, # 1D array of index(intents)
            'idx': indexs  # 1D array of index(id)
        }
        return data


    def label2idx(self, label: str):  # get idx by using label as key
        if label is None:
            return -1
        return self.label_mapping[label]

    def idx2label(self, idx: int):  # get label(intent) by using idx as key
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    # ignore_idx = -100

    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        super(SeqTaggingClsDataset, self).__init__(data, vocab, label_mapping, max_len)


    def collate_fn(self, samples):
        # TODO: implement collate_fn
        tokens = []
        labels = []
        indexs = []
        batch_len = []
        for instance in samples:
            batch_len.append(len(instance['tokens']))
            tokens.append(instance['tokens'])

            tags = instance.get('tags', [None] * self.max_len)
            labels.append([self.label2idx(tags[i]) for i in range(len(tags))])

            indexs.append(instance['id'])
        
        # pad data to same length
        data_after_pad = self.vocab.encode_batch(tokens, self.max_len)
        labels_after_pad = utils.pad_to_len(labels, self.max_len, self.vocab.pad_id)
        # to tensor
        data_after_pad = torch.LongTensor(data_after_pad)
        labels_after_pad = torch.LongTensor(labels_after_pad)
        dict_data = {
            'len': batch_len,
            'data': data_after_pad,
            'label': labels_after_pad,
            'idx': indexs
        }

        return dict_data
