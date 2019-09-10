# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:18:52 2019

@author: s1515896
"""
import os
import csv
import pandas as pd



# change to code dir
dir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/codes/data_process'
os.chdir(dir)
from regex import doc_annotate, read_regex

# change to data dir
dir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/data/stroke/'
os.chdir(dir)


#%% Read and format data
csv.field_size_limit(100000000)
stroke = pd.read_csv("dataWithFullText_utf8.txt", sep='\t', engine="python", encoding="utf-8")   

# Remove invalid records
stroke = stroke[-stroke["ID"].isin([8, 608, 647, 703, 807, 903, 960, 1446, 1707, 1707, 1714, 1716, 1754, 2994, 2995, 2996, 2997, 3943, 4045, 4064, 4066, 4076, 4077, 4083, 3804, 4035])]
stroke.set_index(pd.Series(range(0, len(stroke))), inplace=True)




#%% Output data
stroke.to_csv('rob_stroke_fulltext.txt', sep='\t', encoding='utf-8')
stroke.to_json('rob_stroke_fulltext.json')


#%% Run regex
# Read data
stroke = pd.read_csv("rob_stroke_fulltext.txt", sep='\t', encoding="utf-8")
df = stroke.dropna(subset=['CleanFullText'])
df = df[df["CleanFullText"]!=' ']
df.set_index(pd.Series(range(0, len(df))), inplace=True)

# Read regex string
reg = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/regex/'
regex_randomisation = read_regex(reg+'randomisation.txt')
regex_blinding = read_regex(reg+'blinding.txt')
regex_ssc = read_regex(reg+'ssc.txt')
regex_conflict = read_regex(reg+'conflict.txt')
regex_compliance = read_regex(reg+'compliance.txt')


#list(df.columns)

df['RegexRandomization'] = 0
df['RegexBlinding'] = 0
df['RegexSSC'] = 0
df['RegexConflict'] = 0
df['RegexCompliance'] = 0

# Obtain regex labels 
for i in range(len(df)): 
    df.loc[i,'RegexRandomization'] = doc_annotate(regex_randomisation, df.loc[i,'CleanFullText'])
    df.loc[i,'RegexBlinding'] = doc_annotate(regex_blinding, df.loc[i,'CleanFullText'])
    df.loc[i,'RegexSSC'] = doc_annotate(regex_ssc, df.loc[i,'CleanFullText'])
    df.loc[i,'RegexConflict'] = doc_annotate(regex_conflict, df.loc[i,'CleanFullText'])
    df.loc[i,'RegexCompliance'] = doc_annotate(regex_compliance, df.loc[i,'CleanFullText']) 
    print(i)

# Compute scores
from sklearn.metrics import confusion_matrix
def compute_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1 = 100 * 2*tp / (2*tp+fp+fn)
    accuracy = 100 * (tp+tn) / (tp+tn+fp+fn)
    recall = 100 * tp / (tp+fn)
    specificity = 100 * tn / (tn+fp)
    precision = 100 * tp / (tp+fp)
    print("f1: {0:.2f}% | accuracy: {1:.2f}% | sensitivity: {2:.2f}% | specificity: {3:.2f}% | precision: {4:.2f}%".format(
            f1, accuracy, recall, specificity, precision))
    
compute_score(y_true=df['RandomizationTreatmentControl'], y_pred=df['RegexRandomization'])
compute_score(y_true=df['BlindedOutcomeAssessment'], y_pred=df['RegexBlinding'])
compute_score(y_true=df['SampleSizeCalculation'], y_pred=df['RegexSSC'])
compute_score(y_true=df['ConflictsOfInterest'], y_pred=df['RegexConflict'])
compute_score(y_true=df['AnimalWelfareRegulations'], y_pred=df['RegexCompliance'])






