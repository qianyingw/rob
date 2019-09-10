# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:41:23 2019
ref: github/GokuMohandas/practicalAI: notebooks/12_Embeddings.ipynb
@author: qwang

model1_b: 
    embedding - conv+relu - maxpool - cat - dropout - relu/softmax - fc 
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
        self.fc = nn.Linear(num_channels*len(filter_sizes), num_classes)
        
        if freeze_embeddings:
            self.embeddings.weight.requires_grad = False

    def forward(self, x_in, channel_first=False, apply_softmax=False):
        
        # Embed
        x_in = self.embeddings(x_in)

        # Rearrange input so num_channels is in dim 1 (N, C, L)
        if not channel_first:
            x_in = x_in.transpose(1, 2)
            
        # Conv outputs        
        z = [conv(x_in) for conv in self.conv]
        z = [F.relu(zi) for zi in z]
        z = [F.max_pool1d(zi, zi.size(2)).squeeze(2) for zi in z]
               
        # Concat conv outputs
        z = torch.cat(z, 1)

        # FC layers
        z = self.dropout(z)
#        z = F.relu(z)
        z = F.softmax(z, dim=1)        
        y_pred = self.fc(z)    

        return y_pred