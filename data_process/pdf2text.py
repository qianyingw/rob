# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:37:21 2019
@author: s1515896


convert_pdf_to_txt(path_to_file)
    path_to_file is pdf's full path

convert_multiple(path_df)
    Input dataframe: path_df [Id, path_df] (full paths of pdfs)
    Output dataframe: text_df [Id, text] 
"""

import os

from io import StringIO
import re
import pandas as pd

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

#%% Convert single pdf
def convert_pdf_to_txt(path_to_file):
    if not os.path.exists(path_to_file):
        text = ''
        print('No such file: %s' % path_to_file)
    else:       
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        fp = open(path_to_file, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos=set()
                 
        try:
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
                interpreter.process_page(page)
            text = retstr.getvalue()
            fp.close()
            device.close()
            retstr.close()      
        except:
            text = ''
            print("Cannot convert: " + os.path.basename(path_to_file))
    
    return text


#%% Converts multiple pdfs
# Input dataframe: path_df: [Id, path_df] (full path of pdfs)
# Output dataframe: text_df: [Id, text] 
def convert_multiple(path_df):
    cnum = 0
    pdfnum = 0
    text_df = pd.DataFrame(columns=['Id', 'Text'])
    
    for i in range(len(path_df)):
        pdf_path = path_df.loc[i,'pdf_path']
        if (not os.path.isdir(pdf_path)) and (os.path.splitext(pdf_path)[-1].lower() =='.pdf'):
            pdfnum = pdfnum + 1   
            text = convert_pdf_to_txt(pdf_path)
            text = re.sub(r'\s+', ' ', text)
            
            text_df = text_df.append({'Id': path_df.loc[i,'Id'], 'Text': text}, ignore_index=True)
            if text != '':
                cnum = cnum + 1
    
    print('\n{0} pdfs in total.'.format(pdfnum))            
    print('{0} pdfs has been converted to texts.'.format(cnum))
    return text_df
