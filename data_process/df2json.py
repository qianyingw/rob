# -*- coding: utf-8 -*-
"""
Convert RoB Info df to dict lists
Add 'textTokens' element to each dict list
Output to json

Input: 
    df_info: RoB info dataframe which contains at least following columns
                  'goldID', 'fileLink','DocumentLink','txtLink'.
             For gold data, it is supposed to be
                     df_info[['goldID',
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
   
    json_path: absolute path of final json file

* df_info['txtLink'] are relative paths under 'U:/Datastore/CMVM/scs/groups/DCN/TRIALDEV/CAMARADES/Qianying/RoB/'

Created on Fri Sep 27 11:10:39 2019
@author: qwang
"""

import json

from tqdm import tqdm
from codes.data_process.tokenizer import preprocess_text, tokenize_text


def df2json(df_info, json_path):
   
    dict_list = df_info.to_dict('records')

    # Add fullText and textTokens to each text
    for i, dic in tqdm(enumerate(dict_list)):
        txt_path = dic['txtLink']
        with open(txt_path, 'r', encoding='utf-8') as fp:
            text = fp.read()
        text_processed = preprocess_text(text)        
        # dict_list[i]['fullText'] = text_processed
        dict_list[i]['textTokens'] = tokenize_text(text_processed)
        
    # Covert dictionary list to json
    js = json.dumps(dict_list)
    with open(json_path, 'w') as fout:
        fout.write(js)
    
    
        