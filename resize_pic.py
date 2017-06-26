#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 11:16:30 2017

@author: Lee
"""
import cv2
import numpy as np
def resize_Img(batchsize,src,dst,fx,fy):
    for i in range(1,batchsize+1):
        img=cv2.imread('%s/%d.png'%(src,i),0)
        resize_img=cv2.resize(img,(fx,fy),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('%s/%d.png'%(dst,i),resize_img)
        
resize_Img(6738,"classification","classification",224,224)