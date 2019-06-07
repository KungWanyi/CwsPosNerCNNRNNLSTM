# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:42:53 2019

@author: Gift
"""

import re

f = open('data_pro1.txt', 'r',encoding='utf-8')
fw=open('del_blank.txt', 'a',encoding='utf-8')
p=re.compile("^\s+")

for line in f.readlines():
    s=line.strip()
    #s_str=str(s) 
    pres1 = re.sub(p, '', s)
    
    fw.write(pres1 + '\n')  
fw.close()    
#去掉开头或者结尾空白字符。"^\s+"表示开头空白字符。"\s+$"表示结尾空白字符
#p=re.compile("^\s+")
# 
#str="  jan,increase january,That made good.janttt"
# 
#ss=re.sub(p,'',str)
#print(ss)
