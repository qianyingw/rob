# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:30:06 2019

@author: s1515896
"""

import os
import sys
import pandas as pd
import shutil
from tqdm import tqdm
from time import sleep

# change working directory
wdir = 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/'
os.chdir(wdir)

from codes.data_process.nxml2txt import xml2txt

#%% Copy nxmls from local to S drive
pmc = pd.read_csv("data/silver/pmc/AnimalStudiesFile.csv", sep=',', engine="python", encoding="utf-8")   
list(pmc.columns)
pmc = pmc[['Id', 'ITEM_ID', 'filePath', 'filePathFull', 'score']]
pmc.shape  # (96546, 5)

# Modify filePathFull
pmc['filePathFull'] = pmc['filePathFull'].str.replace('output', 'C:/Users/s1515896/qwang')
pmc['xmlPath'] = pmc['filePathFull'].str.replace('C:/Users/s1515896/qwang', 
                                                 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/data/silver/pmc/nxml')


#%% Copy included nxml files from C drive to S drive
IDFails = []
dat = pmc

for i, row in dat.iterrows():
    inFile = row['filePathFull']
    outFile = row['xmlPath']
    outPath = os.path.dirname(outFile)
    # Create the folder if it doesn't exist
    if os.path.exists(outPath) == False:
        os.makedirs(outPath)
    if os.path.exists(inFile):
        shutil.copy(inFile, outFile)
        print('{}: {} is copied.'.format(i+1, row['ITEM_ID']))
    else:
        IDFails.append(row['Id'])
        print('{} failed!'.format(i+1))
  

#%% Convet nxmls to txts
              
failIds = []
for i, df in pmc.iterrows(): 
    xml_path = df['xmlPath']
    if os.path.exists(xml_path):
        xml2txt(df['xmlPath'], title_included=False, abs_included=False, ack_included=True)
    else:
        failIds.append(df['Id'])
    if (i+1) % 1000 == 0:
        print('{} files have been converted.'.format(i+1))
        
        


    


