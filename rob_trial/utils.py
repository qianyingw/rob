# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:55:35 2019
@author: qwang
"""

#%% Setting
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random
import os
import re
import collections

PATH = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/'
os.chdir(PATH+'data/stroke')

# Parameters
CUDA = False
SEED = 1234
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1


# Check GPU
if torch.cuda.is_available():
    cuda = True

# Set seeds
def set_seed(seed, cuda):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

set_seed(SEED, CUDA)

#%% Data
df = pd.read_csv('rob_stroke_fulltext.txt', sep='\t', encoding='utf-8', index_col=0)
df = df[:1000]
list(df.columns)


df['Text'] = df['CleanFullText']
df['Label'] = df['RandomizationTreatmentControl']

# Create a new dict-like object
by_class = collections.defaultdict(list)
for _, row_df in df.iterrows():
    by_class[row_df['Label']].append(row_df.to_dict())

# Count samples by labels
for c in by_class:
    print("Class {0}: {1}".format(c, len(by_class[c])))
# len(by_class[1])  # Number of samples reporting RoB
# len(by_class[0])  # Number of samples without reporting RoB


# Shuffle and split data by class
final_list = []
for _, doc_list in sorted(by_class.items()):
    np.random.shuffle(doc_list)
    n = len(doc_list)
    n_train = int(TRAIN_SIZE * n)
    n_val = int(VAL_SIZE * n)
    n_test = int(TEST_SIZE * n)
    for doc in doc_list[:n_train]:
        doc['group'] = 'train'
    for doc in doc_list[n_train:(n_train + n_val)]:
        doc['group'] = 'val'
    for doc in doc_list[(n_train + n_val):]:
        doc['group'] = 'test'
    # Add to final_list
    final_list.extend(doc_list)


# df with group labels
final_df = pd.DataFrame(final_list)


# Pre-processing
def preprocess_text(text):
    """
    From https://github.com/bwallace/RoB-CNN/blob/master/data_helpers.py
    Remove the process for abbreviations and punctuations. Spacy can tokenize them.
    """
    text = re.sub(r"[^A-Za-z0-9(),!?.:;\']", " ", text)
    text = text.encode("ascii", errors="ignore").decode()   
    text = re.sub(r'\s+', ' ', text)  # strip whitespaces 
#    text = re.sub(r'\d+', '', text)  # Remove numbers?
    return text.lower()

final_df['Text'] = final_df['Text'].apply(preprocess_text)


# Remove records with empty text 
# It can be empty if all the characters in pre-processing text were non-ascii symbols
for i, text in enumerate(final_df['Text']):
    if not text:
        final_df = final_df.drop([i])


#%% Vocabulary



