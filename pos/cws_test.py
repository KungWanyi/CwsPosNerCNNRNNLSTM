# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:54:41 2019

@author: Gift
"""
import re

if __name__ == '__main__':
    f = open('test2ner.txt', 'r',encoding='utf-8')
    fw=open('cws_test.txt', 'a',encoding='utf-8')



#res="[('药', 'O'), ('物', 'O'), ('选', 'O'), ('择', 'O'), ('目', 'O'), ('前', 'O'), ('主', 'O'), ('要', 'O'), ('根', 'O'), ('据', 'O'), ('癫', 'dis-B'), ('痫', 'dis-I'), ('的', 'O'), ('发', 'O'), ('作', 'O'), ('类', 'O'), ('型', 'O'), ('或', 'O'), ('癫', 'dis-B'), ('痫', 'dis-I'), ('综', 'dis-I'), ('合', 'dis-I'), ('征', 'dis-I'), ('的', 'O'), ('类', 'O'), ('型', 'O'), ('选', 'O'), ('药', 'O'), ('，', 'O'), ('不', 'O'), ('合', 'O'), ('适', 'O'), ('的', 'O'), ('选', 'O'), ('药', 'O'), ('甚', 'O'), ('或', 'O'), ('加', 'O'), ('重', 'O'), ('癫', 'dis-B'), ('痫', 'dis-I'), ('发', 'O'), ('作', 'O'), ('（', 'O'), ('表', 'O'), ('1', 'O'), ('6', 'O'), ('-', 'O'), ('1', 'O'), ('6', 'O'), ('）', 'O'), ('。', 'O')]"
for line in f.readlines():
    res=line.strip()
    res1=str(res)
    res2 = re.sub("\)", '\n', res1)
    #print(res2)
    fw.write(res2 + '\n')  
    
fw.close()  