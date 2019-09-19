# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 09:51:23 2019

@author: qwang
"""

import time
import os
from io import StringIO
import re
import pandas as pd
import numpy as np


import pandas as pd
import os

# change working directory
wdir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/'
os.chdir(wdir)

from codes.data_process.pdf2text import convert_pdf_to_txt, convert_multiple
# from pdf2text import convert_multiple
# from regex import doc_annotate, read_regex


#%% Read and format data
nep = pd.read_csv("data/np/NP_UniqueRecord_ROB.txt", sep='\t', engine="python", encoding="utf-8")
list(nep.columns)
nep['ID'] = np.arange(1, len(nep)+1)

#%% Manual correction
# Replace url/htm/docx link by pdf path
nep.loc[nep.DocumentLink=='http:/europepmc.org/backend/ptpmcrender.fcgi?accid=PMC4251814&blobtype=pdf', 'DocumentLink'] = 'np_fromURL/PubID_400226.pdf'
nep.loc[nep.DocumentLink=='http:/www.degruyter.com/downloadpdf/j/tnsci.2015.6.issue-1/tnsci-2015-0010/tnsci-2015-0010.xml', 'DocumentLink'] = 'np_fromURL/PubID_417813.pdf'
nep.loc[nep.DocumentLink=='http:/www.rehab.research.va.gov/jour/09/46/1/Tan.html', 'DocumentLink'] = 'np_fromURL/PubID_24208.pdf'
nep.loc[nep.DocumentLink=='https:/www.bioscience.org/2012/v4e/af/588/fulltext.php?bframe=PDFII', 'DocumentLink'] = 'np_fromURL/PubID_4352.pdf'
nep.loc[nep.DocumentLink=='https:/www.jstage.jst.go.jp/article/jphs/124/4/124_13249SC/_pdf', 'DocumentLink'] = 'np_fromURL/PubID_420986.pdf'
nep.loc[nep.DocumentLink=='https:/www.researchgate.net/publication/261841186_Anti-Allodynic_and_Anti-hyperalgesic_effects_of_an_ethanolic_extract_and_xylopic_acid_from_the_fruits_of_Xylopia_aethiopica_in_murine_models_of_neuropathic_painAmeya', 'DocumentLink'] = 'np_fromURL/PubID_400489.pdf'
nep.loc[nep.DocumentLink=='https:/www.researchgate.net/publication/260807926_Effect_of_Xylopic_Acid_on_Paclitaxel-induced_Neuropathic_pain_in_rats', 'DocumentLink'] = 'np_fromURL/PubID_400487.pdf'
nep.loc[nep.DocumentLink=='Publications/NP_references/39627_Lynch_2003.docx', 'DocumentLink'] = 'np_fromURL/39627_Lynch_2003.pdf'
nep.loc[nep.DocumentLink=='Publications/NP_references/4368_Palazzo.htm', 'DocumentLink'] = 'np_fromURL/4368_Palazzo.pdf'
# File is old and was replaced by an updated version
nep.loc[nep.DocumentLink=="Publications/NP_references/25171_Fulgenzi_2008.pdf", 'DocumentLink'] = "Publications/NP_references/25171_Fulgenzi_2008_updated.pdf"
nep.loc[nep.DocumentLink=='Publications/NP_references/21016_Chu_2012.pdf', 'DocumentLink'] = 'Publications/NP_references/21016_Chu_2012_updated.pdf'
# Font 'HelveticaNeue-Roman' in abstract can't be read by pdfminer.six and was replaced by txt file
nep.loc[nep.DocumentLink=='Publications/NP_references/4151_Whiteside_2001.pdf', 'DocumentLink'] = 'Publications/NP_references/4151_Whiteside_2001.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/23294_Yamamoto_1995.pdf', 'DocumentLink'] = 'Publications/NP_references/23294_Yamamoto_1995.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/5151_Yamamoto_1996.pdf', 'DocumentLink'] = 'Publications/NP_references/5151_Yamamoto_1996.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/7192_Yamamoto_1999.pdf', 'DocumentLink'] = 'Publications/NP_references/7192_Yamamoto_1999.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/9082_Morse_1997.pdf', 'DocumentLink'] = 'Publications/NP_references/9082_Morse_1997.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/20266_Bennett_1988.pdf', 'DocumentLink'] = 'Publications/NP_references/20266_Bennett_1988.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/1662_Lindner_2000.pdf', 'DocumentLink'] = 'Publications/NP_references/1662_Lindner_2000.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/10959_Mao_1995.pdf', 'DocumentLink'] = 'Publications/NP_references/10959_Mao_1995.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/1339_Marchand_1994.pdf', 'DocumentLink'] = 'Publications/NP_references/1339_Marchand_1994.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/Parks_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/Parks_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/4973_Alba-Delgado_2013.pdf', 'DocumentLink'] = 'Publications/NP_references/4973_Alba-Delgado_2013.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/10709_Hutchinson_2009.pdf', 'DocumentLink'] = 'Publications/NP_references/10709_Hutchinson_2009.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/16780_Xiao_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/16780_Xiao_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/4649_Xiao_2012.pdf', 'DocumentLink'] = 'Publications/NP_references/4649_Xiao_2012.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/22743_Zheng_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/22743_Zheng_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/24165_Cobianchi_2010.pdf', 'DocumentLink'] = 'Publications/NP_references/24165_Cobianchi_2010.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/24935_Ramos_2010.pdf', 'DocumentLink'] = 'Publications/NP_references/24935_Ramos_2010.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/20294_Austin_2010.pdf', 'DocumentLink'] = 'Publications/NP_references/20294_Austin_2010.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/8111_Mika_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/8111_Mika_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/322_Vivoli_2010.pdf', 'DocumentLink'] = 'Publications/NP_references/322_Vivoli_2010.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/16186_Cidral-Filho_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/16186_Cidral-Filho_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/22607_Gwak_2009.pdf', 'DocumentLink'] = 'Publications/NP_references/22607_Gwak_2009.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/24594_Brownjohn_2012.pdf', 'DocumentLink'] = 'Publications/NP_references/24594_Brownjohn_2012.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/14159_Okubo_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/14159_Okubo_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/23457_Kiya_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/23457_Kiya_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/408404_Kiya_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/408404_Kiya_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/18126_Mannelli_2010.pdf', 'DocumentLink'] = 'Publications/NP_references/18126_Mannelli_2010.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/3577_Marinelli_2010.pdf', 'DocumentLink'] = 'Publications/NP_references/3577_Marinelli_2010.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/6961_Ghelardini_2010.pdf', 'DocumentLink'] = 'Publications/NP_references/6961_Ghelardini_2010.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/21866_Chen_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/21866_Chen_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/4366_Miletic_2011.pdf', 'DocumentLink'] = 'Publications/NP_references/4366_Miletic_2011.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/15940_Hermanns_2009.pdf', 'DocumentLink'] = 'Publications/NP_references/15940_Hermanns_2009.txt'
nep.loc[nep.DocumentLink=='Publications/NP_references/26552_Plaza-Villegas_2012.pdf', 'DocumentLink'] = 'Publications/NP_references/26552_Plaza-Villegas_2012.txt'















# Remove records from 'Mental Illness'
mental_list = []
for i, df in nep.iterrows():
    if df['DocumentLink'].split('/')[0] == 'Mental Illness':
        mental_list.append(df['ID'])
nep = nep[-nep["ID"].isin(mental_list)]


# Remove '25548_Detloff_2009.pdf' (a thesis with 187 pages)  
nep = nep[-nep["DocumentLink"].isin(['Publications/NP_references/25548_Detloff_2009.pdf'])]



#%% Modify paths
pdf_folder = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/data/np/npPublications/'
nep['fileLink'] = pdf_folder + nep['DocumentLink'].astype(str)

#len(set(nep['PublicationID']))


#%% Convert pdf to text
path_df = pd.DataFrame(data={'ID': nep['ID'], 'pdf_path': nep['fileLink']})

start = time.time()
df = convert_multiple(path_df, out_path = wdir+'/data/np/npTexts/')
end = time.time()
print('Time elapsed: {} mins.'.format(round((end-start)/60)))


#1697 texts are not NULL.
#1733 pdfs has been converted to texts.
#IDs of files with text length less than 1000:
#[147, 299, 349, 561, 601, 685, 700, 716, 789, 857, 871, 885, 923, 1126, 1152, 1159, 1160, 1192, 1193, 1196, 1199, 1200, 1203, 1212, 1232, 1236, 1316, 1336, 1359, 1367, 1466, 1471, 1481, 1482, 1492, 1495, 1501, 1579, 1600, 1725]
ID_inv1 = [147, 299, 349, 561, 601, 685, 700, 716, 789, 857, 871, 885, 923, 1126, 1152, 1159, 1160, 1192, 1193, 1196, 1199, 1200, 1203, 1212, 1232, 1236, 1316, 1336, 1359, 1367, 1466, 1471, 1481, 1482, 1492, 1495, 1501, 1579, 1600, 1725]

NP = nep
NP = NP[-NP["ID"].isin(ID_inv1)]
# Re-index
NP.set_index(pd.Series(range(0, len(NP))), inplace=True)

#%% Check duplicates
for i, row in NP.iterrows():   
    txt_path = 'data/np/npTexts/np_' + str(row['ID']) + '.txt'
    with open(txt_path, 'r', encoding='utf-8') as fp:
        text = fp.read()
    NP.loc[i,'textLen'] = len(text)


# Remove records with unique length
NP_dup = NP[NP.duplicated(subset=['textLen'], keep=False)]
NP_dup.loc[:,'textSame'] = ''
# Check whether records with same text length are duplicate
dup_grouped = NP_dup.groupby(['textLen'])
dup_grouped = list(dup_grouped)
len(dup_grouped)  # 65

duplen = []  
for i in range(len(dup_grouped)):
    duplen.append(len(dup_grouped[i][1]))
set(duplen)  # {2, 3}



for i, tup in enumerate(dup_grouped):
    if len(tup[1]) == 2:
        df = tup[1]
        df.set_index(pd.Series(range(0,len(df))), inplace=True)
        path0 = 'data/np/npTexts/np_' + str(df['ID'][0]) + '.txt'
        path1 = 'data/np/npTexts/np_' + str(df['ID'][1]) + '.txt'
        with open(path0, 'r', encoding='utf-8') as fp:
            text0 = fp.read()
        with open(path1, 'r', encoding='utf-8') as fp:
            text1 = fp.read()      
        if text0 == text1:
            df['textSame'][0] = 'Yes'

# Convert list to dataframe
frames = [dg[1] for dg in dup_grouped]
dup_df = pd.concat(frames)
dup_df.to_csv('data/np/np_duplicates2.csv', sep=',', encoding='utf-8',
              columns = ['ID', 'textSame', 'textLen', 
                         
                         'RandomizationTreatmentControl',
                         'RandomizationTreatmentControlMethod',
                         'RandomizationModelSham',
                         'RandomizationModelShamMethod',
                         'AllocationConcealment',
                         'AllocationConcealmentMethod',
                         'BlindedOutcomeAssessment',
                         'BlindedOutcomeAssessmentMethod',
                         'SampleSizeCalculation',
                         'SampleSizeCalculationMethod',
                         'Comorbidity',
                         'AnimalWelfareRegulations',
                         'ConflictsOfInterest',
                         'AnimalExclusions',
                         'TypeofDisease',
                         'Project',                         
                         'fileLink','fileLinkNew', 'DocumentLink', 'fileLink'])

# Remove duplicate records 
ID_inv2 = [# duplicates with same labels
              1351,1736,762,1371,1597,
              424,1315,651,1414,1033,1153,
              # duplicates with different labels
              826,1044,662,1666,1587,
              1589,1575,1737,1720,1036,
              847,1689,439,1695,1485,
              1739,1322,1738,124,883,
              330,544,191,1542,1286,
              1287,626,1734]

NP = NP[-NP["ID"].isin(ID_inv2)]

#%% Check long and short texts (not from csv file)
# Read correct texts from txt files, not csv!
print(max(NP['textLen']))
print(min(NP['textLen']))

ID_inv3 = []
for i, row in NP.iterrows():   
    if (row['textLen'] > 100000) or (row['textLen'] < 9000):
        ID_inv3.append(row['ID'])
print('IDs of txts with too long or short texts:\n')
print(ID_inv3)

temp = NP[NP["ID"].isin(ID_inv3)]
temp = temp[['ID', 'textLen', 'DocumentLink']]

# Records with IDs need to be removed (manually checked, see 'np_issues.xlsx')
ID_inv3 = [512,745,914,522,1581,
           429,1554,1553,1038,1067,
           649,1610,922,377,1325]
NP = NP[-NP["ID"].isin(ID_inv3)]
# Re-index
NP.set_index(pd.Series(range(0, len(NP))), inplace=True)


# Recalculate text length
for i, row in NP.iterrows():   
    txt_path = 'data/np/npTexts/np_' + str(row['ID']) + '.txt'
    with open(txt_path, 'r', encoding='utf-8') as fp:
        text = fp.read()
    NP.loc[i,'textLen'] = len(text)

                
                


#%% Output data
dat = NP[['DocumentLink', 'ID', 'textLen',
                    
         'RandomizationTreatmentControl',
         'RandomizationTreatmentControlMethod',
         'RandomizationModelSham',
         'RandomizationModelShamMethod',
         'AllocationConcealment',
         'AllocationConcealmentMethod',
         'BlindedOutcomeAssessment',
         'BlindedOutcomeAssessmentMethod',
         'SampleSizeCalculation',
         'SampleSizeCalculationMethod',
         'Comorbidity',
         'AnimalWelfareRegulations',
         'ConflictsOfInterest',
         'AnimalExclusions',
         
         'TypeofDisease',
         'Project',
         'fileLink',
         'fileExist',
         'copyFlag',
         'fileLinkNew'
         ]]



dat.to_csv('rob_np.txt', sep='\t', encoding='utf-8')
dat.to_csv('rob_np.csv', sep=',', encoding='utf-8')


NP.to_csv('NP.txt', sep='\t', encoding='utf-8')
NP = pd.read_csv("data/np/NP.txt", sep='\t', encoding="utf-8")
max(NP['textLen'])


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







