"""
https://github.com/ahmedbesbes/character-based-cnn/blob/master/src/utils.py
"""

# import os
# import csv
import re
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from tqdm import tqdm


### Stroke data ###
#os.chdir('S:/TRIALDEV/CAMARADES/Qianying/RoB_All')
#csv.field_size_limit(100000000)
#dat = pd.read_csv("datafile/dataWithFullText_utf8.txt", sep='\t', engine="python", encoding="utf-8")   
#dat['RoBText'] = dat['CleanFullText']
#dat['RoBLabel'] = dat['RandomizationTreatmentControl'] 
#dat = dat[-dat["ID"].isin([8, 608, 647, 703, 807, 903, 960, 1446, 1707, 1707, 1714, 1716, 1754, 2994, 
#                           2995, 2996, 2997, 3943, 4045, 4064, 4066, 4076, 4077, 4083, 3804, 4035])]
#dat.set_index(pd.Series(range(0, len(dat))), inplace=True)
#dat.to_csv("datafile/fulldata.csv", sep='\t', encoding='utf-8', index=False)

#data_trial = pd.read_csv("datafile/fulldata.csv", usecols=['RoBText', 'RoBLabel'], sep = '\t', engine = 'python', encoding='utf-8')


#%% text-preprocessing
def remove_urls(text):
    return re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

def remove_non_ascii(text):
    return text.encode("ascii", errors="ignore").decode()

def remove_digits(text):
    return re.sub(r'\d+', '', text)

def remove_punctuations(text):
    return re.sub(r"[!%^&*()=_+{};:$£€@~#|/,.<>?\`\'\"\[\]\\]", " ", text)    # keep hyphen, i.e."-"

def strip_whitespaces(text):
    return re.sub(r'\s+', ' ', text)

def remove_one_character(text):
    return re.sub(r'\b(\w{1})\b', '', text)

def lower(text):
    return text.lower()


preprocessing_setps = {
    'remove_urls': remove_urls,
    'remove_non_ascii': remove_non_ascii,
    'remove_digits': remove_digits,
    'remove_punctuations': remove_punctuations,
    'strip_whitespaces': strip_whitespaces,
    'remove_one_character': remove_one_character,  
    'lower': lower
}

def process_text(steps, text):
    if steps is not None:
        for step in steps:
            text = preprocessing_setps[step](text)
    return text       
#    processed_text = ""
#    for tx in text:
#        processed_text = processed_text + tx + " "
#    return processed_text


#%% custom metrics for evaluation
def rob_metrics(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    output = {}
    
    if 'sensitivity' in list_metrics:
        output['sensitivity'] = metrics.recall_score(y_true, y_pred)
    if 'specificity' in list_metrics:
        output['specificity'] = tn / (tn + fp)   
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)      
    if 'precision' in list_metrics:
        output['precision'] = metrics.precision_score(y_true, y_pred)   
    if 'f1' in list_metrics:
        output['f1'] = metrics.f1_score(y_true, y_pred)
        
    return output

#%% Load data
class rob_data(Dataset):
    def __init__(self, args):

        self.data_path = args.data_path
        self.max_rows = args.max_rows
        self.encoding = args.encoding
        self.chunksize = args.chunksize
        self.preprocessing_steps = args.steps
        self.max_length = args.max_length
        self.num_classes = args.num_classes

        texts, labels = [], []
 
        
        # chunk your dataframes in small portions
        chunks = pd.read_csv(self.data_path, 
                             usecols=[args.text_column, args.label_column],
                             chunksize=self.chunksize,
                             sep = args.sep, engine = 'python', encoding=self.encoding,
                             nrows=self.max_rows)
        
        for df_chunk in tqdm(chunks):
            df_chunk['processed_text'] = (df_chunk[args.text_column].map(lambda text: process_text(self.preprocessing_steps, text)))
            texts += df_chunk['processed_text'].tolist()
            labels += df_chunk[args.label_column].tolist()
            
        print('data loaded successfully with {0} rows'.format(len(labels)))
        self.texts = texts
        self.labels = labels
        self.length = len(self.labels)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        label = self.labels[index]
        raw_text = self.texts[index]
        data = np.asarray(raw_text)
        data = data[np.logical_and(np.logical_and(data!=' ', data!='-'), data!='')]      
        
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = [data.append(None) for i in range(self.max_length-len(data))]
        elif len(data) == 0:       
            data = [None for v in range(self.max_length)] # or ["" for v in range(self.max_length)]    ?
        data = np.array(data)
        return data, label
