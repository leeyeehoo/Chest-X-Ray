#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 12:23:14 2017

@author: Lee
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
def GetTrainData():
    train_data=np.zeros((6738,50176),dtype=np.float32)
    for i in range(1,6739):
        train_data[i-1,:]=np.reshape(plt.imread("classification/%d.png"%(i)),(1,50176))
    return train_data
def GetTrainLabel():
    train_label=np.zeros((6738,1),dtype=np.float32)    
    with open('classification/labels.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for i,row in enumerate(spamreader):            
            if row[1]=='1':
                train_label[i,]=1
            if row[1]=='0':
                train_label[i,]=0                
    return train_label
def shuffle_train():
    new_train=np.concatenate((GetTrainData(),GetTrainLabel()),axis=1)
    a=np.ones((6738,1))
    b=a-GetTrainLabel()
    new_train=np.concatenate((new_train,b),axis=1)
    np.random.shuffle(new_train)
    return new_train

def main():
    print ("hi")
if __name__ == "__main__":
    main()
    gtd=GetTrainData()
    gtl=GetTrainLabel()
    gta=shuffle_train()
    train=gta[0:5615,]
    eval=gta[5615:6738,]
    np.save("nptest/train.npy",train)
    np.save("nptest/eval.npy",eval)