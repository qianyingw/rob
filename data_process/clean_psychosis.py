# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:27:25 2019

@author: s1515896

"""

import pandas as pd
import os

# change to code dir
dir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/codes/data_process'
os.chdir(dir)
from pdf2text import convert_multiple
from regex import doc_annotate, read_regex

# change to data dir
dir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/data/psychosis/'
os.chdir(dir)

#%% Read and format data
psy = pd.read_csv("Psychosis_ROB_categorised.csv", sep=',', engine="python", encoding="utf-8")   

#list(psy.columns)
#['RecordID',
# 'PdfRelativePath',
# 'Phase II Categorization\nRandomisation Reported',
# 'Phase II Categorization\nAllocation Concealment Reported',
# 'Phase II Categorization\nBlinded Assessment of Outcome Reported',
# 'Phase II Categorization\nInclusion/Exclusion Criteria Reported',
# 'Phase II Categorization\nSample Size Calculation Reported',
# 'Phase II Categorization\nConflict of Interest Statement Reported',
# 'Phase II Categorization\nCompliance with Animal Welfare Regulations Reported',
# 'Phase II Categorization\nProtocol Availability Reported']

# Change column names
psy.columns = ['Id', 'PdfPath', 
               'Randomisation', 'AllocationConcealment', 'BlindedAssessmentOutcome', 'InclusionExclusionCriteria', 'SampleSizeCalculation', 'ConflictInterest', 'AnimalWelfare', 'ProtocolAvailability'] 

# Modify paths
psy['PdfPath'] = psy['PdfPath'].str.replace('S:/','U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/')


#%% Convert pdf to text
path_df = pd.DataFrame(data={'Id': psy['Id'], 'pdf_path': psy['PdfPath']})
df = convert_multiple(path_df)


#%% Merge 'NP_UniqueRecord_ROB.txt' and fulltext by Id
left = psy.copy()
right = df.copy()
psy_final = pd.merge(left, right, how='inner', on='Id', validate="one_to_one")

#%% Output data
psy_final.to_csv('rob_psychosis_fulltext.txt', sep='\t', encoding='utf-8')





#%% Run regex
# Read data
psy = pd.read_csv("rob_psychosis_fulltext.txt", sep='\t', encoding="utf-8")
df = psy.dropna(subset=['Text'])
df = df[df["Text"]!=' ']
df.set_index(pd.Series(range(0, len(df))), inplace=True)

# Read regex string
reg = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/regex/'
regex_randomisation = read_regex(reg+'randomisation.txt')
regex_blinding = read_regex(reg+'blinding.txt')
regex_ssc = read_regex(reg+'ssc.txt')
regex_conflict = read_regex(reg+'conflict.txt')
regex_compliance = read_regex(reg+'compliance.txt')

df['RegexRandomization'] = 0
df['RegexBlinding'] = 0
df['RegexSSC'] = 0
df['RegexConflict'] = 0
df['RegexCompliance'] = 0


# Obtain regex labels 
for i in range(len(df)): 
    df.loc[i,'RegexRandomization'] = doc_annotate(regex_randomisation, df.loc[i,'Text'])
    df.loc[i,'RegexBlinding'] = doc_annotate(regex_blinding, df.loc[i,'Text'])
    df.loc[i,'RegexSSC'] = doc_annotate(regex_ssc, df.loc[i,'Text'])
    df.loc[i,'RegexConflict'] = doc_annotate(regex_conflict, df.loc[i,'Text'])
    df.loc[i,'RegexCompliance'] = doc_annotate(regex_compliance, df.loc[i,'Text']) 
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

list(df.columns)
compute_score(y_true=df['Randomisation'], y_pred=df['RegexRandomization'])
compute_score(y_true=df['BlindedAssessmentOutcome'], y_pred=df['RegexBlinding'])
compute_score(y_true=df['SampleSizeCalculation'], y_pred=df['RegexSSC'])
compute_score(y_true=df['ConflictInterest'], y_pred=df['RegexConflict'])
compute_score(y_true=df['AnimalWelfare'], y_pred=df['RegexCompliance'])
