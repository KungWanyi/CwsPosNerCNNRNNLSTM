# -*- coding: utf-8 -*-
"""
Created on Wed May 21 19:30:23 2019

@author: liweimin
"""
import os
import sys

#数据转换
class TransData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.label_dict = {
                      '检查和检验': 'tes',
                      '症状和体征': 'sym',
                      '疾病和诊断': 'dis',
                      '治疗': 'tre',
                      '身体部位': 'bod'}

        self.category_dict ={
                         'O':0,
                         'tre-I': 1,
                         'tre-B': 2,
                         'bod-B': 3,
                         'bod-I': 4,
                         'sym-I': 5,
                         'sym-B': 6,
                         'tes-B': 7,
                         'tes-I': 8,
                         'dis-I': 9,
                         'dis-B': 10,
                         'nt-B' : 11,
                         'nt-I' : 12,
                         'nr-B' : 13,
                         'nr-I' : 14
                        }
        self.origin_path = os.path.join(cur, 'data_origin')
        self.train_filepath = os.path.join(cur, 'train.txt')
        return

#trans实现
    def trans(self):
        f = open(self.train_filepath, 'w+', encoding='UTF-8')
        count = 0
        for root,dirs,files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)
                if 'original' not in filepath:
                    continue
                label_filepath = filepath.replace('.txtoriginal','')
                print(filepath, '\t\t', label_filepath)
                content = open(filepath,'r', encoding='UTF-8').read().strip()
                res_dict = {}
                for line in open(label_filepath,'r', encoding='UTF-8'):
                    res = line.strip().split('	')
                    start = int(res[1])
                    end = int(res[2])
                    label = res[3]
                    label_id = self.label_dict.get(label)
                    for i in range(start, end+1):
                        if i == start:
                            label_category = label_id + '-B'
                        else:
                            label_category = label_id + '-I'
                        res_dict[i] = label_category

                for indx, char in enumerate(content):
                    char_label = res_dict.get(indx, 'O')
                    print(char, char_label)
                    f.write(char + '\t' + char_label + '\n')
        f.close()
        return



if __name__ == '__main__':
    processor = TransData()
    train_datas = processor.trans()