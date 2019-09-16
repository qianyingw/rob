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



#%% Read data
csv.field_size_limit(100000000)
stroke = pd.read_csv("dataWithFullText_utf8.txt", sep='\t', engine="python", encoding="utf-8")   
list(stroke.columns)


#%% Check file extension and text length
for i, df in stroke.iterrows():
    if df['fileLink'].lower().endswith('.pdf') == False:
        print(df['ID'], df['DocumentLink'])

stroke.loc[stroke.ID==610, 'DocumentLink'] = 'qw_newDownload/Ref Id 217 Schabitz.pdf'
stroke.loc[stroke.ID==611, 'DocumentLink'] = 'qw_newDownload/12644 Tanaka R.txt'
stroke.loc[stroke.ID==612, 'DocumentLink'] = 'qw_newDownload/6058 Tatlisumak T.txt'
stroke.loc[stroke.ID==613, 'DocumentLink'] = 'qw_newDownload/6181 Kawamata T.pdf'
stroke.loc[stroke.ID==614, 'DocumentLink'] = 'qw_newDownload/6285 Bethel A.txt'
stroke.loc[stroke.ID==615, 'DocumentLink'] = 'qw_newDownload/329 Bernaudin M.pdf'
stroke.loc[stroke.ID==616, 'DocumentLink'] = 'qw_newDownload/5595 Hayashi T.pdf'
stroke.loc[stroke.ID==1147, 'DocumentLink'] = 'qw_newDownload/Martin 2009.pdf'
stroke.loc[stroke.ID==1446, 'DocumentLink'] = 'qw_newDownload/4843.txt'

stroke.loc[stroke.ID==1706, 'DocumentLink'] = 'qw_newDownload/Garcia, O 2410.pdf'
stroke.loc[stroke.ID==1707, 'DocumentLink'] = 'qw_newDownload/Garcia, O 2594.pdf'
stroke.loc[stroke.ID==1714, 'DocumentLink'] = 'qw_newDownload/Hernandez-Fonseca, K 2307.pdf'
stroke.loc[stroke.ID==1716, 'DocumentLink'] = 'qw_newDownload/Hewett, SJ 2914 full text page.txt'

stroke.loc[stroke.ID==2583, 'DocumentLink'] = 'qw_newDownload/Zhang et al 1996 full text.txt'
stroke.loc[stroke.ID==2593, 'DocumentLink'] = 'qw_newDownload/Escott et al 1998 full text.pdf'
stroke.loc[stroke.ID==2597, 'DocumentLink'] = 'qw_newDownload/Iadecola et al 1996.txt'
stroke.loc[stroke.ID==2602, 'DocumentLink'] = 'qw_newDownload/Nagayama et al 1998.pdf'
stroke.loc[stroke.ID==2607, 'DocumentLink'] = 'qw_newDownload/Stagliano et al 1997 full text.pdf'


stroke.loc[stroke.ID==4248, 'DocumentLink'] = 'qw_newDownload/3369 Waxham.pdf'
stroke.loc[stroke.ID==4249, 'DocumentLink'] = 'qw_newDownload/3420 Maeda.pdf'

#610 Publications/Behavioural pdfs/BDNF/Ref Id 217 Schabitz.mht
#611 Publications/Behavioural pdfs/BFGF/12644 Tanaka R.htm
#612 Publications/Behavioural pdfs/BFGF/6058 Tatlisumak T.htm
#613 Publications/Behavioural pdfs/BFGF/6181 Kawamata T.html
#614 Publications/Behavioural pdfs/BFGF/6285 Bethel A.htm
#615 Publications/Behavioural pdfs/EPO/329 Bernaudin M.mht
#616 Publications/Behavioural pdfs/VEGF/5595 Hayashi T.mht
#1147 Publications/HD/Martin 2009.doc
#1446 Publications/Lacunar/4843.mht

#1706 Publications/MK801/Garcia, O 2410.mht
#1707 Publications/MK801/Garcia, O 2594.mht
#1714 Publications/MK801/Hernandez-Fonseca, K 2307.mht
#1716 Publications/MK801/Hewett, SJ 2914 full text page.mht

#2583 Publications/NOS donors/Zhang et al 1996 full text.htm
#2593 Publications/NOS Inhibitors/Escott et al 1998 full text.htm
#2597 Publications/NOS Inhibitors/Iadecola et al 1996.htm
#2602 Publications/NOS Inhibitors/Nagayama et al 1998.htm
#2607 Publications/NOS Inhibitors/Stagliano et al 1997 full text.htm

#2997 Publications/Psoriasis/Pdfs/868.mht  # Removed later
#4035 Publications/SCI+StemCells/445_Zhang_2010.doc   # Removed later
#4248 Publications/transgenic/All TG PDFs/TG article new/3369 Waxham.mht
#4249 Publications/transgenic/All TG PDFs/TG article new/3420 Maeda.mht  



#%% Check duplicates
stroke['textLen'] = stroke['CleanFullText'].apply(lambda x: len(x))
# Remove records with unique length
stroke_dup = stroke[stroke.duplicated(subset=['textLen'], keep=False)]
stroke_dup['textSame'] = ''
# Check whether records with same text length are duplicate
dup_grouped = stroke_dup.groupby(['textLen'])
dup_grouped = list(dup_grouped)

len(dup_grouped)  # 100  
duplen = []  
for i in range(len(dup_grouped)):
    duplen.append(len(dup_grouped[i][1]))
set(duplen)  # {2, 3}


for i, tup in enumerate(dup_grouped):
    if len(tup[1]) == 2:
        df = tup[1]
        df.set_index(pd.Series(range(0,len(df))), inplace=True)
        if df['CleanFullText'][0] == df['CleanFullText'][1]:
            df['textSame'][0] = 'Yes'



# Convert list to dataframe
frames = [dg[1] for dg in dup_grouped]
dup_df = pd.concat(frames)
dup_df.to_csv('issues.csv', sep=',', encoding='utf-8',
              columns = ['ID', 'textSame', 'textLen', 
                         'RandomizationTreatmentControl',
                         'AllocationConcealment',
                         'BlindedOutcomeAssessment',
                         'SampleSizeCalculation',
                         'AnimalExclusions',
                         'Comorbidity',
                         'AnimalWelfareRegulations',
                         'ConflictsOfInterest',
                         'DocumentLink', 'fileLink'])

        
    
    
df_checklen = stroke[['ID', 'textLen', 'DocumentLink']]


#87  # Text too long (567 pages)
#853  # No fulltext
#2997  # Chinese paper, no English fulltext
#4035  # Chinese paper, no English fulltext
#9845  # Chinese paper, no English fulltext
#859  # Chinese paper, no English fulltext

#%% 

# Remove invalid records

InvalidIDs = [87, 853, 859, 2997, 4035, 9845, 
              # duplicates with same labels
              4170, 1390, 1408, 1395, 985, 
              764, 958, 961, 4163, 2579,
              306, 3935, 118, 1380, 4168,              
              2072, 876, 3926, 3997, 270,
              877, 4221, 62, 3994, 1036, 
              2708, 1017, 2080,
              # duplicates with different labels

              ]





stroke = stroke[-stroke["ID"].isin(InvalidIDs)]
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






