# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:10:41 2019

@author: s1515896
"""

import os
path = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/'
os.chdir(path+'datafile')



from argparse import Namespace
import collections
#from collections import Counter

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import csv
#import copy
#import json
#import string

import torch
#from torch.utils.data import Dataset, DataLoader
#import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
#from gensim.models.keyedvectors import KeyedVectors

#%% ==================================== Set up ====================================
# Set Numpy and PyTorch seeds
def set_seeds(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

## Random setting for DataLoader
#def _init_fn(seed):
#    np.random.seed(seed)
 
# cudnn setting
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True    
      
# Creating directories
def create_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
# Arguments
args = Namespace(
    seed=1234,
    cuda=True,
    shuffle=True,
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir=path+'codes/rnn_20190626',
    train_size=0.80,
    val_size=0.10,
    test_size=0.10,
    pretrained_embeddings=None,
    cutoff=25, # minimum document count
    num_epochs=20,
    early_stopping_criteria=5,
    learning_rate=1e-3,
    batch_size=64,
    max_seq_len=5000,
    embedding_dim=200,
    rnn_hidden_dim=128,
    hidden_dim=100,
    num_layers=1,
    bidirectional=False,
    dropout_p=0.5,
)

# Set seeds
set_seeds(seed=args.seed, cuda=args.cuda)

# Create save dir
create_dirs(args.save_dir)

# Expand filepaths
args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
args.model_state_file = os.path.join(args.save_dir, args.model_state_file)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")
print("Using CUDA: {}".format(args.cuda))

#%% ==================================== Data ====================================
csv.field_size_limit(100000000)

df = pd.read_csv("fulldata.csv", usecols=['text', 'label_random', 'label_blind', 'label_ssz'], sep = '\t', engine = 'python', encoding='utf-8')
df.loc[df.label_random==1, 'label'] = 'random'
df.loc[df.label_random==0, 'label'] = 'non-random'
#df.loc[df.label_blind==1, 'label'] = 'blinded'
#df.loc[df.label_blind==0, 'label'] = 'non-blinded'
#df.loc[df.label_ssz==1, 'label'] = 'ssz'
#df.loc[df.label_ssz==0, 'label'] = 'non-ssz'
df.label.value_counts()

# Split by category
by_label = collections.defaultdict(list)
for _, row in df.iterrows():
    by_label[row.label].append(row.to_dict())
for label in by_label:
    print("{0}: {1}".format(label, len(by_label[label])))
    
    
# Create split data
final_list = []
for _, item_list in sorted(by_label.items()):
    if args.shuffle:
        np.random.shuffle(item_list)
    n = len(item_list)
    n_train = int(args.train_size*n)
    n_val = int(args.val_size*n)
    n_test = int(args.test_size*n)

  # Give data point a split attribute
    for item in item_list[:n_train]:
        item['split'] = 'train'
    for item in item_list[n_train:n_train+n_val]:
        item['split'] = 'val'
    for item in item_list[n_train+n_val:]:
        item['split'] = 'test'  

    # Add to final list
    final_list.extend(item_list)    

# df with split datasets
split_df = pd.DataFrame(final_list)
split_df["split"].value_counts()

# Preprocessing
def preprocess_text(text):
    text = ' '.join(word.lower() for word in text.split(" ")) 
    text = re.sub(r"([.,;!?])", r" \1 ", text)     # Match a single character present in the list 
    text = re.sub(r"[^a-zA-Z.,;!?-]+", r" ", text) # Match a single character not present in the list
    return text

#def preprocess_text(text):
#    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
#    text = text.encode("ascii", errors="ignore").decode()
#    text = re.sub(r'\d+', '', text)
#    text = re.sub(r"[!%^&*()=_+{};:$£€@~#|/,.<>?\`\'\"\[\]\\]", " ", text)  # [!%^&*()=_+{};:$£€@~#|/<>?\`\'\"\[\]\\]
#    text = re.sub(r'\b(\w{1})\b', '', text) 
#    text = re.sub(r'\s+', ' ', text)
#    text.lower()
#    return text


#preprocess_text(split_df.text[3608])
    

split_df.text = split_df.text.apply(preprocess_text)
# Remove records with null text
for i in range(len(split_df)):
    if not split_df['text'][i]:
        print(i)
        split_df = split_df.drop([i])