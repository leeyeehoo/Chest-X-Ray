#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:00:15 2017

@author: Lee
"""
import json
import csv

def write_Out(descript):
    csvfile = file('classification/labels.csv', 'wb')
    writer = csv.writer(csvfile)
    writer.writerows(descript)

with open('image/data_new.json', 'r') as f:
    data = json.load(f)
test_data=[]
for i in range(1,6739):
    if data['%d'%i]['items']==data['1']['items']:
        test_data.append([i,0])
    else:
        test_data.append([i,1])
write_Out(test_data)
