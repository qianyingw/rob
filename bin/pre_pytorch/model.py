# -*- coding: utf-8 -*-
"""
Convolutional Neural Networks for Document Classification
==============================
Ref: 1) https://github.com/Shawn1993/cnn-text-classification-pytorch
     2) https://github.com/yongjincho/cnn-text-classification-pytorch
     3) https://github.com/ahmedbesbes/character-based-cnn/blob/master/src/cnn_model.py

torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)

torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    Input:  (batch size, in_channels, height, width) = (N, C_in, H, W)
    Output: (batch_size, out_channels, height_out, width_out) = (N, C_out, H_out, W_out)

torch.squeeze(input, dim=None, out=None) → Tensor
    Returns a tensor with all the dimensions of input of size 1 removed.
    For example, if input is of shape (A×1×B×C×1×D) then the out tensor will be of shape (A×B×C×D).
    
torch.unsqueeze(input, dim, out=None) → Tensor
    Returns a new tensor with a dimension of size one inserted at the specified position.
    A dim value within the range [-input.dim() - 1, input.dim() + 1) can be used. 
    Negative dim will correspond to unsqueeze() applied at dim = dim + input.dim() + 1.
==============================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models.keyedvectors import KeyedVectors



word_vectors = KeyedVectors.load_word2vec_format('wikipedia-pubmed-and-PMC-w2v.bin', binary=True)
embed_dim = 200
embed_matrix = np.zeros((vocab_size, embed_dim))

num_words = 5000
for word, i in word_index.items():
    if i>=num_words:
        continue
    try:
        embedding_vector = word_vectors[word] # vector shape (200, 1)
        embed_matrix[i] = embedding_vector[:embed_dim]
    except KeyError:
        embed_matrix[i]=np.random.normal(0, np.sqrt(0.25), embed_dim)

del(word_vectors)



def create_embed_layer(embed_matrix, not_trainable=False):
    num_embed, embed_dim = embed_matrix.size()
    embed_layer = nn.Embedding(num_embed, embed_dim)
    embed_layer.load_state_dict({'weight': embed_matrix})
    if not_trainable:
        embed_layer.weight.requires_grad = False
    return embed_layer
    

class DocCNN(nn.Module):

    def __init__(self, args):
        super(DocCNN, self).__init__()
        
        self.args = args
        vocab_size = args.vocab_size
        embed_dim = args.embed_dim
        num_filter = args.num_filter
        filter_size = args.filter_size
        num_class = args.num_class
                
        self.embedding = nn.Embedding(vocab_size, embed_dim)        
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filter, (f, embed_dim)) for f in filter_size])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(num_filter*len(filter_size), num_class)

    def init_weights(self, pretrained_word_vectors, is_static=False):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False
        
    def forward(self, x):
        # x.shape = [batch_size, doc_len, vocab_size]?
        x = self.embedding(x)           # [batch_size, doc_len, embed_dim)]
        x = torch.unsqueeze(x, 1)       # [batch_size, in_channels=1, doc_len, embed_dim]

        x_list = []     
        for conv in self.convs:
            xi = F.relu(conv(x))               # [batch_size, num_filter, new_doc_len, 1]
            xi = torch.squeeze(xi, 3)          # [batch_size, num_filter, new_doc_len]
            xi = F.max_pool1d(xi, xi.size(2))  # [batch_size, num_filter, 1]
            xi = torch.squeeze(xi, 2)          # [batch_size, num_filter]
            x_list.append(xi)
        x = torch.cat(x_list, 1)               # [batch_size, num_filter*len(filter_size)]
        
        x = self.dropout(x)                    # [batch_size, num_filter*len(filter_size)]
        logits = self.fc(x)                    # [batch_size, num_class]
        
        probs = F.softmax(logits)              # [batch_size, num_class]
        classes = torch.max(probs, 1)[1]       # [batch_size]
    
        return probs, classes
    

doccnn = DocCNN()
print(doccnn)
        


