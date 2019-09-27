# -*- coding: utf-8 -*-
"""

Created on Mon Aug 19 10:27:25 2019
@author: qwang

"""


import time
import os
import pandas as pd
import numpy as np


# change working directory
wdir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/'
os.chdir(wdir)

from codes.data_process.pdf2text import convert_multiple
from codes.data_process.df2json import df2json
# from codes.data_process.regex import doc_annotate, read_regex
# from codes.data_process.tokenizer import preprocess_text, tokenize_text


#%% Read and format data
psy = pd.read_csv("data/psychosis/Psychosis_ROB_categorised.csv", sep=',', engine="python", encoding="utf-8")   
list(psy.columns)
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
psy.columns = ['RecordID', 'fileLink', 
               
               'RandomizationTreatmentControl',
               'AllocationConcealment',
               'BlindedOutcomeAssessment',
               'AnimalExclusions',
               'SampleSizeCalculation',
               'ConflictsOfInterest',
               'AnimalWelfareRegulations',
               'ProtocolAvailability'] 

psy['ID'] = np.arange(1, len(psy)+1)

# Modify paths
psy['DocumentLink'] = psy['fileLink'].str.replace('S:/JISC Analytics Lab/CAMARADES Datasets/', '')


# Manual correction
psy.loc[psy.DocumentLink=='All_psychosis/11288_1997.pdf', 'DocumentLink'] = 'All_psychosis/11288_1997.txt'
psy.loc[psy.DocumentLink=='All_psychosis/6641_1994.pdf', 'DocumentLink'] = 'All_psychosis/6641_1994.txt'
psy.loc[psy.DocumentLink=='All_psychosis/8460_1996.pdf', 'DocumentLink'] = 'All_psychosis/8460_1996.txt'
psy.loc[psy.DocumentLink=='All_psychosis/4533_2010.pdf', 'DocumentLink'] = 'All_psychosis/4533_2010.txt'
psy.loc[psy.DocumentLink=='All_psychosis/11965_2010.pdf', 'DocumentLink'] = 'All_psychosis/11965_2010.txt'
psy.loc[psy.DocumentLink=='All_psychosis/10350_2009.pdf', 'DocumentLink'] = 'All_psychosis/10350_2009.txt'
psy.loc[psy.DocumentLink=='All_psychosis/10605_2010.pdf', 'DocumentLink'] = 'All_psychosis/10605_2010.txt'
psy.loc[psy.DocumentLink=='All_psychosis/14530_2010.pdf', 'DocumentLink'] = 'All_psychosis/14530_2010.txt'
psy.loc[psy.DocumentLink=='All_psychosis/257_2012.pdf', 'DocumentLink'] = 'All_psychosis/257_2012.txt'
psy.loc[psy.DocumentLink=='All_psychosis/14512_2012.pdf', 'DocumentLink'] = 'All_psychosis/14512_2012.txt'
psy.loc[psy.DocumentLink=='All_psychosis/13061_2010.pdf', 'DocumentLink'] = 'All_psychosis/13061_2010.txt'
psy.loc[psy.DocumentLink=='All_psychosis/11398_2011.pdf', 'DocumentLink'] = 'All_psychosis/11398_2011.txt'
psy.loc[psy.DocumentLink=='All_psychosis/13524_2004.pdf', 'DocumentLink'] = 'All_psychosis/13524_2004_updated.pdf'


# Modify pdf path
pdf_folder = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/data/psychosis/'
psy['fileLink'] = pdf_folder + psy['DocumentLink'].astype(str)




#%% Convert pdf to text
path_df = pd.DataFrame(data={'ID': psy['ID'], 'pdf_path': psy['fileLink']})
start = time.time()
df = convert_multiple(path_df, out_path = wdir+'/data/psychosis/psyTexts/')
end = time.time()
print('Time elapsed: {} mins.'.format(round((end-start)/60)))
#    2400 texts are not NULL.
#    2465 pdfs has been converted to texts.
#    IDs of files with text length less than 1000:
#    
#    [69, 116, 125, 158, 232, 379, 469, 511, 557, 558, 624, 637, 669, 681, 712, 726, 760, 789, 802, 804, 889, 936, 938, 958, 1016, 1034, 1041, 1052, 1072, 1093, 1228, 1279, 1532, 1548, 1561, 1580, 1618, 1673, 1826, 1827, 1860, 1864, 1877, 1879, 1902, 1903, 1904, 1950, 2040, 2055, 2056, 2105, 2113, 2215, 2228, 2254, 2282, 2287, 2311, 2320, 2343, 2370, 2371, 2411, 2428]
#    Time elapsed: 254 mins.

ID_inv1 = [69, 116, 125, 158, 232, 379, 469, 511, 557, 558, 624, 637, 669, 681, 712, 726, 760, 789, 802, 804, 889, 936, 938, 958, 1016, 1034, 1041, 1052, 1072, 1093, 1228, 1279, 1532, 1548, 1561, 1580, 1618, 1673, 1826, 1827, 1860, 1864, 1877, 1879, 1902, 1903, 1904, 1950, 2040, 2055, 2056, 2105, 2113, 2215, 2228, 2254, 2282, 2287, 2311, 2320, 2343, 2370, 2371, 2411, 2428]
psy = psy[-psy["ID"].isin(ID_inv1)]
# Re-index
psy.set_index(pd.Series(range(0, len(psy))), inplace=True)



#%% Check duplicates
for i, row in psy.iterrows():   
    txt_path = 'data/psychosis/psyTexts/psy_' + str(row['ID']) + '.txt'
    with open(txt_path, 'r', encoding='utf-8') as fp:
        text = fp.read()
    psy.loc[i,'textLen'] = len(text)


# Remove records with unique length
psy_dup = psy[psy.duplicated(subset=['textLen'], keep=False)]
psy_dup.loc[:,'textSame'] = ''
# Check whether records with same text length are duplicate
dup_grouped = psy_dup.groupby(['textLen'])
dup_grouped = list(dup_grouped)
len(dup_grouped)  # 56

duplen = []  
for i in range(len(dup_grouped)):
    duplen.append(len(dup_grouped[i][1]))
set(duplen)  # {2, 3}


for i, tup in enumerate(dup_grouped):
    if len(tup[1]) == 2:
        df = tup[1]
        df.set_index(pd.Series(range(0,len(df))), inplace=True)
        path0 = 'data/psychosis/psyTexts/psy_' + str(df['ID'][0]) + '.txt'
        path1 = 'data/psychosis/psyTexts/psy_' + str(df['ID'][1]) + '.txt'
        with open(path0, 'r', encoding='utf-8') as fp:
            text0 = fp.read()
        with open(path1, 'r', encoding='utf-8') as fp:
            text1 = fp.read()      
        if text0 == text1:
            df['textSame'][0] = 'Yes'

# Convert list to dataframe
frames = [dg[1] for dg in dup_grouped]
dup_df = pd.concat(frames)

dup_df.to_csv('data/psychosis/psy_duplicates.csv', sep=',', encoding='utf-8',
              columns = ['ID', 'textSame', 'textLen',                        
                         'RandomizationTreatmentControl',
                         'AllocationConcealment',
                         'BlindedOutcomeAssessment',
                         'AnimalExclusions',
                         'SampleSizeCalculation',
                         'ConflictsOfInterest',
                         'AnimalWelfareRegulations',
                         'ProtocolAvailability',                                   
     
                    'fileLink', 'DocumentLink', 'RecordID'])

# No duplicate records for psychosis data!


    
#%% Check long and short texts (not from csv file)
# Read correct texts from txt files, not csv!
print(max(psy['textLen']))  # 322167
print(min(psy['textLen']))  # 1015

ID_inv2 = []
for i, row in psy.iterrows():   
    if (row['textLen'] > 100000) or (row['textLen'] < 9000):
        ID_inv2.append(row['ID'])
print('IDs of txts with too long or short texts:\n')
print(ID_inv2)

temp = psy[psy["ID"].isin(ID_inv2)]
temp = temp[['ID', 'textLen', 'DocumentLink']]


# Records with IDs need to be removed (manually checked, see 'psy_issues.xlsx')
ID_inv3 = [578, 850, 70]
psy = psy[-psy["ID"].isin(ID_inv3)]
# Re-index
psy.set_index(pd.Series(range(0, len(psy))), inplace=True)


# Recalculate text length
for i, row in psy.iterrows():   
    txt_path = 'data/psychosis/psyTexts/psy_' + str(row['ID']) + '.txt'
    with open(txt_path, 'r', encoding='utf-8') as fp:
        text = fp.read()
    psy.loc[i,'textLen'] = len(text)

print(max(psy['textLen']))  # 121983
print(min(psy['textLen']))  # 12136

#%% Output data
# Add columns
psy['txtLink'] = 'data/psychosis/psyTexts/psy_' + psy['ID'].astype(str) + '.txt'
psy['goldID'] = 'psy' + psy['ID'].astype(str)  # ID for all the gold data

psy.to_csv('data/psychosis/rob_psychosis_info.txt', sep='\t', encoding='utf-8')
list(psy.columns)

#['RecordID',
# 'fileLink',
# 'RandomizationTreatmentControl',
# 'AllocationConcealment',
# 'BlindedOutcomeAssessment',
# 'AnimalExclusions',
# 'SampleSizeCalculation',
# 'ConflictsOfInterest',
# 'AnimalWelfareRegulations',
# 'ProtocolAvailability',
# 'ID',
# 'DocumentLink',
# 'textLen',
# 'txtLink',
# 'goldID']


#%% Tokenization to json file
psy = pd.read_csv("data/psychosis/rob_psychosis_info.txt", sep='\t', engine="python", encoding="utf-8", index_col = 0)   

df = psy[['goldID',
          'fileLink',
          'DocumentLink',
          'txtLink',
          'RandomizationTreatmentControl',
          'AllocationConcealment',
          'BlindedOutcomeAssessment',
          'SampleSizeCalculation',
          'AnimalWelfareRegulations',
          'ConflictsOfInterest',
          'AnimalExclusions']]


df2json(df_info = df, json_path = 'data/psychosis/rob_psychosis_fulltokens.json')






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
