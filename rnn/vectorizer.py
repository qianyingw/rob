# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:13:44 2019

@author: s1515896
"""


import numpy as np
import json

from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

#import data_process
#from data_process import _init_fn

# Random setting for DataLoader
def _init_fn(seed):
    np.random.seed(seed)
    
#%% ==================================== Vocabulary ====================================
class Vocabulary(object):
    def __init__(self, token_to_idx=None):

        # Token to index
        if token_to_idx is None:
            token_to_idx = {}
        self.token_to_idx = token_to_idx

        # Index to token
        self.idx_to_token = {idx: token \
                             for token, idx in self.token_to_idx.items()}

    def to_serializable(self):
        return {'token_to_idx': self.token_to_idx}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_tokens(self, tokens):
        return [self.add_token[token] for token in tokens]

    def lookup_token(self, token):
        if token not in self.token_to_idx:
            raise KeyError("the token (%s) is not in the Vocabulary" % token)
        return self.token_to_idx[token]

    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self.idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self.token_to_idx)

# Test: Vocabulary instance
#label_vocab = Vocabulary()
#for index, row in df.iterrows():
#    label_vocab.add_token(row.label)
#print(label_vocab) # __str__
#print(len(label_vocab)) # __len__
#index = label_vocab.lookup_token("random")
#print(index)
#print(label_vocab.lookup_index(index))


#%% ==================================== Sequence vocabulary ====================================
class SequenceVocabulary(Vocabulary):
    def __init__(self, token_to_idx=None, unk_token="<UNK>",
                 mask_token="<MASK>", begin_seq_token="<BEGIN>",
                 end_seq_token="<END>"):

        super(SequenceVocabulary, self).__init__(token_to_idx)

        self.mask_token = mask_token
        self.unk_token = unk_token
        self.begin_seq_token = begin_seq_token
        self.end_seq_token = end_seq_token

        self.mask_index = self.add_token(self.mask_token)
        self.unk_index = self.add_token(self.unk_token)
        self.begin_seq_index = self.add_token(self.begin_seq_token)
        self.end_seq_index = self.add_token(self.end_seq_token)
        
        # Index to token
        self.idx_to_token = {idx: token \
                             for token, idx in self.token_to_idx.items()}

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update({'unk_token': self.unk_token,
                         'mask_token': self.mask_token,
                         'begin_seq_token': self.begin_seq_token,
                         'end_seq_token': self.end_seq_token})
        return contents

    def lookup_token(self, token):
        return self.token_to_idx.get(token, self.unk_index)
    
    def lookup_index(self, index):
        if index not in self.idx_to_token:
            raise KeyError("the index (%d) is not in the SequenceVocabulary" % index)
        return self.idx_to_token[index]
    
    def __str__(self):
        return "<SequenceVocabulary(size=%d)>" % len(self.token_to_idx)

    def __len__(self):
        return len(self.token_to_idx)


## Test: Get word counts
#word_counts = Counter()
#for text in split_df.text:
#    for token in text.split(" "):
#        if token not in string.punctuation:
#            word_counts[token] += 1
#
## Test: Create SequenceVocabulary instance
#paper_vocab = SequenceVocabulary()
#for word, word_count in word_counts.items():
#    if word_count >= args.cutoff:
#        paper_vocab.add_token(word)
#print(paper_vocab) # __str__
#print(len(paper_vocab)) # __len__
#index = paper_vocab.lookup_token("general")
#print(index)
#print(paper_vocab.lookup_index(index))


#%% ==================================== Vectorizer ====================================
class PapersVectorizer(object):
    def __init__(self, paper_vocab, label_vocab):
        self.paper_vocab = paper_vocab
        self.label_vocab = label_vocab

    def vectorize(self, paper):
        indices = [self.paper_vocab.lookup_token(token) for token in paper.split(" ")]
        indices = [self.paper_vocab.begin_seq_index] + indices + [self.paper_vocab.end_seq_index]
        
        # Create vector
        paper_length = len(indices)
        vector = np.zeros(paper_length, dtype=np.int64)
        vector[:len(indices)] = indices
        return vector, paper_length
    
    def unvectorize(self, vector):
        tokens = [self.paper_vocab.lookup_index(index) for index in vector]
        paper = " ".join(token for token in tokens)
        return paper

    @classmethod
    def from_dataframe(cls, df, cutoff):
        
        # Create class vocab
        label_vocab = Vocabulary()        
        for label in sorted(set(df.label)):
            label_vocab.add_token(label)

        # Get word counts
        word_counts = Counter()
        for paper in df.text:
            for token in paper.split(" "):
                word_counts[token] += 1
        
        # Create paper vocab
        paper_vocab = SequenceVocabulary()
        for word, word_count in word_counts.items():
            if word_count >= cutoff:
                paper_vocab.add_token(word)
        
        return cls(paper_vocab, label_vocab)

    @classmethod
    def from_serializable(cls, contents):
        paper_vocab = SequenceVocabulary.from_serializable(contents['paper_vocab'])
        label_vocab = Vocabulary.from_serializable(contents['label_vocab'])
        return cls(paper_vocab=paper_vocab, label_vocab=label_vocab)
    
    def to_serializable(self):
        return {'paper_vocab': self.paper_vocab.to_serializable(),
                'label_vocab': self.label_vocab.to_serializable()}

## Test: Vectorizer instance
#vectorizer = PapersVectorizer.from_dataframe(split_df, cutoff=args.cutoff)
#print(vectorizer.paper_vocab)
#print(vectorizer.label_vocab)
#vectorized_paper, paper_length = vectorizer.vectorize(preprocess_text("Roger Federer wins the Wimbledon tennis tournament."))
#print(np.shape(vectorized_paper))
#print("paper_length:", paper_length)
#print(vectorized_paper)
#print(vectorizer.unvectorize(vectorized_paper))


#%% ==================================== Dataset class ====================================
class PapersDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer

        # Data splits
        self.train_df = self.df[self.df.split=='train']
        self.train_size = len(self.train_df)
        self.val_df = self.df[self.df.split=='val']
        self.val_size = len(self.val_df)
        self.test_df = self.df[self.df.split=='test']
        self.test_size = len(self.test_df)
        self.lookup_dict = {'train': (self.train_df, self.train_size), 
                            'val': (self.val_df, self.val_size),
                            'test': (self.test_df, self.test_size)}
        self.set_split('train')

        # Class weights (for imbalances)
        class_counts = df.label.value_counts().to_dict()
        def sort_key(item):
            return self.vectorizer.label_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)

    @classmethod
    def load_dataset_and_make_vectorizer(cls, df, cutoff):
        train_df = df[df.split=='train']
        return cls(df, PapersVectorizer.from_dataframe(train_df, cutoff))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, df, vectorizer_filepath):
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)
        return cls(df, vectorizer)

    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            return PapersVectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(self.vectorizer.to_serializable(), fp)

    def set_split(self, split="train"):
        self.target_split = split
        self.target_df, self.target_size = self.lookup_dict[split]

    def __str__(self):
        return "<Dataset(split={0}, size={1})".format(self.target_split, self.target_size)

    def __len__(self):
        return self.target_size

    def __getitem__(self, index):
        row = self.target_df.iloc[index]
        paper_vector, paper_length = self.vectorizer.vectorize(row.text)
        label_index = self.vectorizer.label_vocab.lookup_token(row.label)
        return {'paper': paper_vector, 
                'paper_length': paper_length, 
                'label': label_index}

    def get_num_batches(self, batch_size):
        return len(self) // batch_size

    def generate_batches(self, batch_size, collate_fn, shuffle=True, drop_last=False, device="cpu"):
        dataloader = DataLoader(dataset=self, batch_size=batch_size,
                                collate_fn=collate_fn, shuffle=shuffle, 
                                drop_last=drop_last,
                                num_workers = 0,
                                worker_init_fn=_init_fn)
        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = data_dict[name].to(device)
            yield out_data_dict


## Test: Dataset instance
#dataset = PapersDataset.load_dataset_and_make_vectorizer(df=split_df, cutoff=args.cutoff)
##dataset.save_vectorizer(args.vectorizer_file)
##vectorizer = PapersDataset.load_vectorizer_only(vectorizer_filepath=args.vectorizer_file)
#print(dataset) # __str__
#input_ = dataset[5] # __getitem__
#print(input_['paper'], input_['paper_length'], input_['label'])
#print(dataset.vectorizer.unvectorize(input_['paper']))
#print(dataset.class_weights) # tensor([0.0003, 0.0011])