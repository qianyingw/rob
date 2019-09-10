# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:14:20 2019

@author: s1515896
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



#%% ==================================== Model ====================================
def gather_last_relevant_hidden(hiddens, x_lengths):
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(hiddens[batch_index, column_index])
    return torch.stack(out)

class PapersModel(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, rnn_hidden_dim, 
                 hidden_dim, output_dim, num_layers, bidirectional, dropout_p, 
                 pretrained_embeddings=None, freeze_embeddings=False, 
                 padding_idx=0):
        super(PapersModel, self).__init__()
        
        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                           num_embeddings=num_embeddings,
                                           padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embeddings = nn.Embedding(embedding_dim=embedding_dim,
                                           num_embeddings=num_embeddings,
                                           padding_idx=padding_idx,
                                           _weight=pretrained_embeddings)
        
        # Conv weights
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=rnn_hidden_dim, 
                          num_layers=num_layers, batch_first=True, 
                          bidirectional=bidirectional)
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_hidden_dim, 
                          num_layers=num_layers, batch_first=True, 
                          bidirectional=bidirectional)
        
     
        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(rnn_hidden_dim, output_dim)
        
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

    def forward(self, x_in, x_lengths, apply_softmax=False):
        
        # Embed
        x_in = self.embeddings(x_in)
            
        # Feed into RNN
        # out: [seq_len, batch, num_directions*hidden_size]; output features h_t from the last layer of the GRU/LSTM, for each t
        # h_n: [num_layers*num_directions, batch, hidden_size]; hidden state for t = seq_len
        # c_n: [num_layers*num_directions, batch, hidden_size]; cell state for t = seq_len.
        out, h_n = self.gru(x_in)         
        out, (h_n, c_n) = self.lstm(x_in)
        
        # Gather the last relevant hidden state
        out = gather_last_relevant_hidden(out, x_lengths)

        # FC layers
        z = self.dropout(out)
        y_pred = self.fc(z)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred