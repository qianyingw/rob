# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:15:40 2019

@author: s1515896
"""

import re
import nltk
#import os
#dir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/regex'
#os.chdir(dir)



#%% 

# Read regex string
def read_regex(regex_path):
    f = open(regex_path, "r")
    regex = f.read()
    f.close()
    return regex

# Document annotation
def doc_annotate(regex, doc):
    match = re.search(regex, doc)
    if match:
        doc_label = 1
    else:
        doc_label = 0
    return doc_label


# Sentence tokenisation and annotation
def sent_annotate(regex, doc):
    sent_label = []
    sent_list = nltk.sent_tokenize(doc)
    for i, sent in enumerate(sent_list):
        match = re.search(regex, sent)
        if match:
            sent_label.append([sent, 1])
        else:
            sent_label.append([sent, 0])
    return sent_label

#%%
#regex_randomisation = read_regex('randomisation.txt')
#regex_blinding = read_regex('blinding.txt')
#regex_ssc = read_regex('ssc.txt')
#regex_conflict = read_regex('conflict.txt')
#regex_compiance = read_regex('compliance.txt')
#
#
##%% test
#test_str = "When calculating the average latency, the cut-off time was assigned to the normal responses. The average latency was taken as ameasure for the severity of cold allodynia; shorter tail withdrawal latency was interpreted as more severe allodynia. All behavioral tests were performed in blinded fashion."
#sent_labeled = sent_annotate(regex_blinding, test_str)
#sent_labeled[0]    # [sentence, label]
#sent_labeled[0][0] # sentence
#sent_labeled[0][1] # label