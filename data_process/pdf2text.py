# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:37:21 2019
@author: s1515896


convert_pdf_to_txt(path_to_file)
    path_to_file is pdf's full path

convert_multiple(path_df)
    Input dataframe: path_df [ID, path_df] (full paths of pdfs)
    Output dataframe: text_df [ID, text] 
"""

import os

from io import StringIO
import re
import pandas as pd

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

import pikepdf
import PyPDF2


#%% Convert single pdf
def convert_pdf_to_txt(path_to_file):
    
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()
    
    if os.path.exists(path_to_file):
        fp = open(path_to_file, 'rb')   
        try:
            reader = PyPDF2.PdfFileReader(fp)
            if reader.isEncrypted == True:
                pdf = pikepdf.open(path_to_file)
                new_path = path_to_file.split('.pdf')[0]+'_extractable.pdf'
                pdf.save(new_path)
                fp = open(new_path, 'rb')  
        except:
            # Skip 'PdfReadError: EOF marker not found'. It's a bug hasn't been fixed by PyPDF2. 
            print("Skip encryption check.")
                                  
        try:   
            for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
                interpreter.process_page(page)  
            text = retstr.getvalue()                          
        except:
            text = ''
            print('Cannot be converted: {}.'.format(path_to_file))
        fp.close()
        device.close()
        retstr.close()  
            
    else:  
        text = ''
        print('No such file: %s' % path_to_file)
    
    return text


#%% Converts multiple pdfs
# Input dataframe: path_df: [ID, path_df] (full path of pdfs)
# Output dataframe: text_df: [ID, text] 
def convert_multiple(path_df, out_path):
    cnum = 0
    pdfnum = 0
    pdf_fail = []
    text_df = pd.DataFrame(columns=['ID', 'Text'])
    
    for i, row in path_df.iterrows():
        pdf_path = row['pdf_path']
        
        # Read txt file directly
        if pdf_path.lower().endswith('.txt'):
            with open(pdf_path, 'r') as fp:
                text = fp.read()
             
        if os.path.splitext(pdf_path)[-1].lower() =='.pdf':
            pdfnum = pdfnum + 1              
            text = convert_pdf_to_txt(pdf_path)
            text = re.sub(r'\s+', ' ', text)            
                                    
        if len(text) < 1000:
            text = ''
        text_df = text_df.append({'ID': row['ID'], 'Text': text}, ignore_index=True)
                      
        if text != '':
            cnum = cnum + 1
            out_filename = out_path + 'psy_{}.txt'.format(row['ID'])
            with open(out_filename, 'w', encoding="utf-8") as fout:
                fout.write(text)  
        else:
            pdf_fail.append(row['ID'])
        
        print(row['ID'])
    
    
    print('\n{0} texts are not NULL.'.format(cnum))            
    print('{0} pdfs has been converted to texts.'.format(pdfnum))
    if len(pdf_fail) != 0:
        print('IDs of files with text length less than 1000:\n')
        print(pdf_fail)
   
    return text_df
