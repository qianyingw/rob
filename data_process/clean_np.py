# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:51:23 2019

@author: s1515896
"""


import os
from io import StringIO
import re
import pandas as pd
import numpy as np


import pandas as pd
import os

# change to code dir
dir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/codes/data_process'
os.chdir(dir)
from pdf2text import convert_multiple
from regex import doc_annotate, read_regex

# change to data dir
dir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/data/np/'
os.chdir(dir)

#%% Read and format data
np = pd.read_csv("NP_UniqueRecord_ROB.txt", sep='\t', engine="python", encoding="utf-8")

# Replace url/htm/docx link by pdf path
np.loc[np.DocumentLink=='http:/europepmc.org/backend/ptpmcrender.fcgi?accid=PMC4251814&blobtype=pdf', 'DocumentLink'] = 'np_fromURL/PubID_400226.pdf'
np.loc[np.DocumentLink=='http:/www.degruyter.com/downloadpdf/j/tnsci.2015.6.issue-1/tnsci-2015-0010/tnsci-2015-0010.xml', 'DocumentLink'] = 'np_fromURL/PubID_417813.pdf'
np.loc[np.DocumentLink=='http:/www.rehab.research.va.gov/jour/09/46/1/Tan.html', 'DocumentLink'] = 'np_fromURL/PubID_24208.pdf'
np.loc[np.DocumentLink=='https:/www.bioscience.org/2012/v4e/af/588/fulltext.php?bframe=PDFII', 'DocumentLink'] = 'np_fromURL/PubID_4352.pdf'
np.loc[np.DocumentLink=='https:/www.jstage.jst.go.jp/article/jphs/124/4/124_13249SC/_pdf', 'DocumentLink'] = 'np_fromURL/PubID_420986.pdf'
np.loc[np.DocumentLink=='https:/www.researchgate.net/publication/261841186_Anti-Allodynic_and_Anti-hyperalgesic_effects_of_an_ethanolic_extract_and_xylopic_acid_from_the_fruits_of_Xylopia_aethiopica_in_murine_models_of_neuropathic_painAmeya', 'DocumentLink'] = 'np_fromURL/PubID_400489.pdf'
np.loc[np.DocumentLink=='https:/www.researchgate.net/publication/260807926_Effect_of_Xylopic_Acid_on_Paclitaxel-induced_Neuropathic_pain_in_rats', 'DocumentLink'] = 'np_fromURL/PubID_400487.pdf'
np.loc[np.DocumentLink=='Publications/NP_references/39627_Lynch_2003.docx', 'DocumentLink'] = 'np_fromURL/39627_Lynch_2003.pdf'
np.loc[np.DocumentLink=='Publications/NP_references/4368_Palazzo.htm', 'DocumentLink'] = 'np_fromURL/4368_Palazzo.pdf'

np['Id'] = np['PublicationID']

# Modify paths
np['DocumentLink'] = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/data/np/npPublications/' + np['DocumentLink'].astype(str)



#%% Convert pdf to text
path_df = pd.DataFrame(data={'Id': np['Id'], 'pdf_path': np['DocumentLink']})
df = convert_multiple(path_df[0:10])


#%% Merge 'NP_UniqueRecord_ROB.txt' and with fulltext by Id

# Remove duplicates and 2333 left (needs to be checked by Jing; shouldn't have any duplicate records)
left = dat.copy()
right = df.copy()
left = left.drop_duplicates(subset='DocumentLink', keep='first')
right = right.drop_duplicates(subset='DocumentLink', keep='first')
np_final = pd.merge(left, right, how='inner', on='DocumentLink', validate="one_to_one")

# Remove records without full-text and 2314 left
#np_final['Text'].replace('', np.nan, inplace=True)
#np_final.dropna(subset=['Text'], inplace=True)


#%% Output data
np_final.to_csv('rob_np_fulltext.txt', sep='\t', encoding='utf-8')



#%% Run regex
# Read data
np = pd.read_csv("rob_np_fulltext.txt", sep='\t', encoding="utf-8")
df = np.dropna(subset=['Text'])
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
compute_score(y_true=df['RandomizationTreatmentControl'], y_pred=df['RegexRandomization'])
compute_score(y_true=df['BlindedOutcomeAssessment'], y_pred=df['RegexBlinding'])
compute_score(y_true=df['SampleSizeCalculation'], y_pred=df['RegexSSC'])
#compute_score(y_true=df['ConflictInterest'], y_pred=df['RegexConflict'])
#compute_score(y_true=df['AnimalWelfare'], y_pred=df['RegexCompliance'])







#%% Text processing

temp1 = temp.lower()
temp1 = text.strip('\n')
temp1 = text.replace('\n',' ')
temp2 = re.sub(r'\s+', ' ', text)

def remove_before_abs(text):
#    re.sub(r'.*?abstract', 'abstract', text, count=1)
    if len(text.rsplit('abstract'))>1:
        t = text.rsplit('abstract')[0]  # texts before the first occurence of 'abstract'
        return text[len(t):]
    else:
        return text

def remove_after_ref(text):
    if len(text.rsplit('references'))>1:
        t = text.rsplit('references')[-1]  # texts after the last occurence of 'references'
        return text[:-(len(t)+len('references_'))]
    else:
        return text



dirpath = 'C:\Users\gputman\Desktop\Control_File_Tracker\Input\\'
output = 'C:\Users\gputman\Desktop\Control_File_Tracker\Output\New Microsoft Excel Worksheet.csv'
csvout = pd.DataFrame()

for filename in files:
    data = pd.read_csv(filename, sep=':', index_col=0, header=None).T
    csvout = csvout.append(data)

csvout.to_csv(output)







