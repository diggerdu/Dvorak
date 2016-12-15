# -*- coding: utf-8 -*-

import os
import pickle


train_dir = "train"
eva_dir = "eva"

train_dict = pickle.load(open("train_dict","rb"))
'''
for file in os.listdir(train_dir):
    if train_dict.get(file.strip()) is None and not "DS" in file:
        train_dict.update({file.strip():"萝卜头"})
        print file
'''
for file in train_dict.keys():
    if file == "huzc_20160510_robotou_6.wav":
        print file
    if file.strip() not in os.listdir(train_dir):
        print file

pickle.dump(train_dict,open("train_dict","wb"))
