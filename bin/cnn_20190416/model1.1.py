# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:15:01 2019
ref: github/GokuMohandas/practicalAI: notebooks/12_Embeddings.ipynb
@author: qwang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PapersModel(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_input_channels, 
                 num_channels, hidden_dim, num_classes, dropout_p, filter_sizes,
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
        self.conv = nn.ModuleList([nn.Conv1d(num_input_channels, num_channels, 
                                             kernel_size=f) for f in filter_sizes])
     
        # FC weights
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(num_channels*len(filter_sizes), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

    def forward(self, x_in, channel_first=False, apply_softmax=False):
        
        # Embed
        x_in = self.embeddings(x_in)

        # Rearrange input so num_channels is in dim 1 (N, C, L)
        if not channel_first:
            x_in = x_in.transpose(1, 2)

#        # Conv outputs
#        z1 = F.relu(self.conv[0](x_in))
#        z1 = F.max_pool1d(z1, z1.size(2)).squeeze(2)
#        z2 = F.relu(self.conv[1](x_in))
#        z2 = F.max_pool1d(z2, z2.size(2)).squeeze(2)
#        z3 = F.relu(self.conv[2](x_in))
#        z3 = F.max_pool1d(z3, z3.size(2)).squeeze(2)
#        # Concat conv outputs
#        z = torch.cat([z1, z2, z3], 1)
            
        # Conv outputs        
        z = [conv(x_in) for conv in self.conv]
        z = [F.relu(zi) for zi in z]
        z = [F.max_pool1d(zi, zi.size(2)).squeeze(2) for zi in z]
               
        # Concat conv outputs
        z = torch.cat(z, 1)

        # FC layers
        z = self.dropout(z)
        z = self.fc1(z)
        y_pred = self.fc2(z)
        
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred