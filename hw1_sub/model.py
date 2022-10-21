from typing import Dict
import logging

import torch
from torch import nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        **kwargs
    ) -> None:

        super(SeqClassifier, self).__init__()
    
        # TODO: model architecture
        for key, value in kwargs.items():
            setattr(self, key, value) 

        if self.rnn_method == 'RNN':
            self.model = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer,
                                   bidirectional=self.bidirectional, nonlinearity='relu', dropout=self.drop_rate)
        elif self.rnn_method == 'GRU':
            self.model = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer,
                                   bidirectional=self.bidirectional, dropout=self.drop_rate)
        elif self.rnn_method == 'LSTM':
            self.model = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer,
                                   bidirectional=self.bidirectional, dropout=self.drop_rate)

        self.separator = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_size * 2, self.output_size)
        )
        
        self.embedding()
        
    
    def embedding(self):
        self.embed = nn.Sequential(
            Embedding.from_pretrained(self.embeddings, freeze=False, padding_idx=self.pad_id),
            nn.LayerNorm(len(self.embeddings[0]))
        ).to(self.device)


    def forward(self, x, embed_type=None) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embeddings = self.embed(x).permute(1, 0, 2)
        if self.rnn_method == 'LSTM':
            result, (hiddens, _) = self.model(embeddings, None)
        else:
            result, hiddens = self.model(embeddings, None)

        if self.bidirectional:
            final_layer = hiddens.reshape(self.num_layer, 2, -1, self.hidden_size)[-1]
            logging.debug(final_layer.shape)
            feature = final_layer.permute(1, 0, 2).reshape(-1, 2 * self.hidden_size)
        else:
            feature = hiddens[-1]

        logging.debug(f'output feature shape= {feature.shape}')
        probability = self.separator(feature)
        return probability


class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        **kwargs
    ) -> None:

        super(SeqTagger, self).__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.rnn_method == 'RNN':
            self.model = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer,
                                   bidirectional=self.bidirectional, nonlinearity='relu', dropout=self.drop_rate)
        elif self.rnn_method == 'GRU':
            self.model = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer,
                                   bidirectional=self.bidirectional, dropout=self.drop_rate)
        elif self.rnn_method == 'LSTM':
            self.model = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layer,
                                   bidirectional=self.bidirectional, dropout=self.drop_rate)
        
        self.separator = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 2),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.hidden_size * 2, self.output_size)
        )

        self.embedding()
    
    def embedding(self):
        self.embed = nn.Sequential(
            Embedding.from_pretrained(self.embeddings, freeze=False, padding_idx=self.pad_id),
            nn.LayerNorm(len(self.embeddings[0]))
        ).to(self.device)

    def forward(self, x, embed_type=None):
        # TODO: implement model forward
        embeddings = self.embed(x).permute(1, 0, 2)
        if self.rnn_method == 'LSTM':
            result, (hiddens, _) = self.model(embeddings, None)
        else:
            result, hiddens = self.model(embeddings, None)
        probability = self.separator(result).permute(1, 2, 0)
        return probability
