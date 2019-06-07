# -*- coding: utf-8 -*-
"""
Created on Wed May 28 21:11:28 2019

@author: liweimin
"""
import os
import sys
import re
 
# make English text clean
def clean_text(text):
    # keep English, digital and space
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub('', text)
 
# make Chinese text clean
#def clean_zh_text(text):
#    # keep English, digital and Chinese
#    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
#    return comp.sub('', text)
 
    


if __name__ == '__main__':
#    text_en = '(How old are you? Could you give me your pen)'
#    text_zh = '$你好！我是个程序猿，标注码农￥'
#    print(clean_en_text(text_en))
#    print(clean_zh_text(text_zh)) 
#打开ner文件进行处理，并写入data_pro中    
    f = open('cws_test.txt', 'r',encoding='utf-8')
    fw=open('data_pro.txt', 'a',encoding='utf-8')
#
#
for line in f.readlines():
    s=line.strip()
#    #pre_w=ner.predict(s)
#    #prew=str(pre_w)
#    prew=clean_text(s)
#    fw.write(prew + '\n')
    
#删除无关符号 
    #s="[('药', 'O'), ('物', 'O'), ('选', 'O'), ('择', 'O'), ('目', 'O'), ('前', 'O'), ('主', 'O'), ('要', 'O'), ('根', 'O'), ('据', 'O'), ('癫', 'dis-B'), ('痫', 'dis-I'), ('的', 'O'), ('发', 'O'), ('作', 'O'), ('类', 'O'), ('型', 'O'), ('或', 'O'), ('癫', 'dis-B'), ('痫', 'dis-I'), ('综', 'dis-I'), ('合', 'dis-I'), ('征', 'dis-I'), ('的', 'O'), ('类', 'O'), ('型', 'O'), ('选', 'O'), ('药', 'O'), ('，', 'O'), ('不', 'O'), ('合', 'O'), ('适', 'O'), ('的', 'O'), ('选', 'O'), ('药', 'O'), ('甚', 'O'), ('或', 'O'), ('加', 'O'), ('重', 'O'), ('癫', 'dis-B'), ('痫', 'dis-I'), ('发', 'O'), ('作', 'O'), ('（', 'O'), ('表', 'O'), ('1', 'O'), ('6', 'O'), ('-', 'O'), ('1', 'O'), ('6', 'O'), ('）', 'O'), ('。', 'O')]"
    s_str=str(s) 
    pres1 = re.sub("[[(),O'']", '', s_str)
    pres2 = re.sub("[]]", '', pres1)
#pres2 = re.sub('['']', '', pres1)

#print ("处理后: ", pres2+'\n')    
    fw.write(pres2 + '\n')  
    
fw.close()    
    
    
    
    
    
    
    
    