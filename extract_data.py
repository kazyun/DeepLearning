#! /usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import numpy as np
import pandas as pd
#from openpyxl import load_workbook
from gensim.models import word2vec
import xlrd
import logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


reload(sys)
sys.setdefaultencoding('utf8')

file_path = os.path.abspath('.')
xlsx_file = xlrd.open_workbook( file_path + '/oldFile_hitProducts.xlsx' )
model = word2vec.Word2Vec.load( file_path + "/cookpad_v2.model")

#data = pd.read_excel( file_path + '/oldFile_hitProducts.xlsx', encoding="cp932")

sheet_name = xlsx_file.sheet_names()[0]

sheet = xlsx_file.sheet_by_index(0)

rows = sheet.row_values(1)
cols = sheet.col_values(3)

word = sheet.row( 1 )[4].value
word.encode( 'utf-8' )

unknow_words = open( file_path + "/test1.txt" , "w")

#print model[u'スフレ']
'''
word2 = sheet.row(1)[11].value
word2.encode('utf-8')
if len( word2 ) <= 0:
    print "ok"

print model[word]
'''



#print len( sheet.row(1) )
#print model[word].shape[0]


for line in range(1, 87):

    temp_word = []

    for col in range( 3, 33):

        words = sheet.row(line)[col].value


        words.encode( 'utf-8' )


        if len( words ) <= 0:

            break

        if words not in model.wv.vocab:

            unknow_words.write( words + " " )
            temp_word.append( words )
            #print words

    if len( temp_word ) != 0:
        unknow_words.write( "\n" )
