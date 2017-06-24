#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 09:36:48 2017

@author: Lee
"""

import numpy as np
def shuffle_train():
    new_train=np.load("nptest/train.npy")
    np.random.shuffle(new_train)
    return new_train
def shuffle_eval():
    new_eval=np.load("nptest/eval.npy")
    np.random.shuffle(new_eval)
    return new_eval
