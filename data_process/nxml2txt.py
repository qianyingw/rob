# -*- coding: utf-8 -*-
"""
Convert nxml to txt (for single PMC nxml file)

Created on Fri Sep 27 11:55:57 2019
@author: qwang
"""

import os
import re
import xml.etree.ElementTree as ET  
  

#%% Extract text from single xml
def parse(root, elem_string):
    for elem in root.findall(elem_string):
        text = ET.tostring(elem, encoding="utf-8", method="xml").decode("utf-8")
        text = re.sub("</title>", ". ", text)          
        text = re.sub("[\<].*?[\>]", " ", text) # {.*?} matches any character (except for line terminators) between '<' and '>'
        text = re.sub(r"\s+", " ", text)
        return text.lstrip()  


def xml2txt(xml_path, title_included=False, abs_included=False, ack_included=True):
      
    tree = ET.parse(xml_path)
    root = tree.getroot() 
        
    title = parse(root, elem_string="front/article-meta/title-group")
    abstract = parse(root, elem_string="front/article-meta/abstract") 
    body = parse(root, elem_string="body")
    ack = parse(root, elem_string="back/ack")
    
    text = body
    if abs_included and abstract:
        text = abstract + ' ' + text
    if title_included and title:
        text = title + '. ' + text
    if ack_included and ack:
        text = text + ' ' + ack  
         
    txt_path = os.path.dirname(xml_path) + '/' + re.sub('.nxml', '.txt', os.path.basename(xml_path))
    
    with open(txt_path, 'w', encoding='utf-8') as fout:
        fout.write(text)



#%% (My experiment) Extract text from single xml
#    os.chdir('M:/qwang/data_PMC/my_trial')
#    tree = ET.parse('PMC3614260.nxml')  
#    tree = ET.parse('journal.pmed.0020162.xml')
#    root = tree.getroot()
#    
#    # Abstract
#    for elem in root.findall("front/article-meta/abstract"):
#        abstract = ET.tostring(elem, encoding="utf-8", method="xml").decode("utf-8")  
#        abstract = re.sub("[\<].*?[\>]", " ", abstract) # {.*?} matches any character (except for line terminators) between '<' and '>'
#        abstract = re.sub(r"\s+", " ", abstract)
#        abstract = abstract.lstrip()  
#    
#    # Main body part
#    for elem in root.findall("body"):
#        body = ET.tostring(elem, encoding="utf-8", method="xml").decode("utf-8")  
#        body = re.sub("[\<].*?[\>]", " ", body) # {.*?} matches any character (except for line terminators) between '<' and '>'
#        body = re.sub(r"\s+", " ", body)
#        body = body.lstrip()  
#    
#    # Acknowledgement
#    for elem in root.findall("back/ack"):
#        ack = ET.tostring(elem, encoding="utf-8", method="xml").decode("utf-8")  
#        ack = re.sub("[\<].*?[\>]", " ", ack) # {.*?} matches any character (except for line terminators) between '<' and '>'
#        ack = re.sub(r"\s+", " ", ack)
#        ack = ack.lstrip()  
#        
#    text = body + ' ' + ack
#    
#    with open('try.txt','w') as f:
#        f.write(text)
#    f.close()

