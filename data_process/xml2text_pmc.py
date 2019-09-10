# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:30:06 2019

@author: s1515896
"""

import os
import pandas as pd


import xml.etree.ElementTree as ET  
import re
import time



#%% Extract text from single xml
def parse(root, elem_string):
    for elem in root.findall(elem_string):
        text = ET.tostring(elem, encoding="utf-8", method="xml").decode("utf-8")
        text = re.sub("</title>", ". ", text)          
        text = re.sub("[\<].*?[\>]", " ", text) # {.*?} matches any character (except for line terminators) between '<' and '>'
        text = re.sub(r"\s+", " ", text)
        return text.lstrip()  


def single_xml2txt(xml_path, txt_path, title_included=False, abs_included=False, ack_included=True):

    tree = ET.parse(xml_path)
    root = tree.getroot() 
    
    title = parse(root, elem_string="front/article-meta/title-group")
    abstract = parse(root, elem_string="front/article-meta/abstract") 
    body = parse(root, elem_string="body")
    ack = parse(root, elem_string="back/ack")
    
    text = body
    if abs_included and abs:
        text = abstract + ' ' + text
    if title_included and title:
        text = title + '. ' + text
    if ack_included and ack:
        text = text + ' ' + ack
      
    with open(txt_path,'w') as f:
        f.write(text)
    f.close()


#single_xml2txt(xml_path='M:/qwang/data_PMC/my_trial/journal.pmed.0020162.xml', 
#               txt_path='M:/qwang/data_PMC/my_trial/journal.pmed.0020162.txt',
#               title_included=True, 
#               abs_included=True, 
#               ack_included=True)


#%% Extract text from multiple xml files
# path_df is a dataframe including two columns: Id, xml_path (full path of XML file)
# txt_folder is a string

def multiple_xml2txt(path_df, txt_folder): 
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)   
    path_df['txt_path'] = '' #txt_folder + path_df['Id'].astype(str) + '.txt'      
#    issue_df = pd.DataFrame(columns = ['Id', 'xml_path'])
    
    for i in range(len(path_df)):
        try:
            single_xml2txt(xml_path=path_df.loc[i,'xml_path'],    # 'xml_path'
                           txt_path=path_df.loc[i,'txt_path'],    # 'txt_path'
                           title_included=False, 
                           abs_included=False, 
                           ack_included=True)
            path_df.loc[i, 'txt_path'] = txt_folder + path_df.loc[i,'Id'].astype(str) + '.txt'
        except:
#            issue_df = issue_df.append({'Id': path_df.loc[i,'Id'], 'xml_path': path_df.loc[i,'xml_path']}, ignore_index=True)
            print("[%s" % path_df.loc[i,'xml_path'] + "] can't be converted or doesn't exist!") 
          
    path_df.to_csv(txt_folder+"animal_txt_path.csv", sep=',', encoding='utf-8', index=False)
#    issue_df.to_csv(txt_folder+"animal_problem.csv", sep=',', encoding='utf-8', index=False)   
    print('Finish!')



#AnimalStudiesFile = pd.read_csv("M:/qwang/data_PMC/output/oa_bulk/AnimalStudiesFile.csv", usecols=['score', 'ITEM_ID', 'Id', 'filePathFull', 'textfilename'], sep=',', engine="python", encoding="utf-8")
path_df = pd.DataFrame(data={'Id': [111, 222, 333], 'xml_path': ['M:/qwang/data_PMC/my_trial/PMC3614260.nxml', 'M:/qwang/data_PMC/my_trial/PMC3614262.nxml', 'M:/qwang/data_PMC/my_trial/journal.pmed.0020162.xml']})

animal_file = pd.read_csv("M:/qwang/data_PMC/output/oa_bulk/AnimalStudiesFile.csv", usecols=['Id', 'filePathFull'], sep=',', engine="python", encoding="utf-8")
animal_file['xml_path'] = 'M:/qwang/data_PMC/' + animal_file['filePathFull']
path_df = pd.DataFrame(data={'Id': animal_file['Id'], 'xml_path': animal_file['xml_path']}) # 632,446


# df1
df1 = path_df.loc[0:1000,:]
start_time = time.time()
multiple_xml2txt(path_df = df1, txt_folder='M:/qwang/data_PMC/output_animal_txt/1/')
print("Elapsed time: %.1f" % ((time.time()-start_time)/60) + " minutes.\n" + "="*50 + "\n")


    
#%% (My experiment) Extract text from single xml
os.chdir('M:/qwang/data_PMC/my_trial')
tree = ET.parse('PMC3614260.nxml')  
tree = ET.parse('journal.pmed.0020162.xml')
root = tree.getroot()

# Abstract
for elem in root.findall("front/article-meta/abstract"):
    abstract = ET.tostring(elem, encoding="utf-8", method="xml").decode("utf-8")  
    abstract = re.sub("[\<].*?[\>]", " ", abstract) # {.*?} matches any character (except for line terminators) between '<' and '>'
    abstract = re.sub(r"\s+", " ", abstract)
    abstract = abstract.lstrip()  

# Main body part
for elem in root.findall("body"):
    body = ET.tostring(elem, encoding="utf-8", method="xml").decode("utf-8")  
    body = re.sub("[\<].*?[\>]", " ", body) # {.*?} matches any character (except for line terminators) between '<' and '>'
    body = re.sub(r"\s+", " ", body)
    body = body.lstrip()  

# Acknowledgement
for elem in root.findall("back/ack"):
    ack = ET.tostring(elem, encoding="utf-8", method="xml").decode("utf-8")  
    ack = re.sub("[\<].*?[\>]", " ", ack) # {.*?} matches any character (except for line terminators) between '<' and '>'
    ack = re.sub(r"\s+", " ", ack)
    ack = ack.lstrip()  
    
text = body + ' ' + ack

with open('try.txt','w') as f:
    f.write(text)
f.close()



