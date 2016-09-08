#!/usr/bin/env python
# coding=utf-8
import os
import mfcc
import numpy as np
import scipy.io.wavfile as wav

audio_path = "audio"
suffix = ".list"


li_list = list()
for li in os.listdir(audio_path):
    if li.endswith(suffix):
        li_list.append(li.split('.')[0])

for li in li_list:
    cur_dir = audio_path + '/' + li.split('_')[0] + li.split('_')[1] + '/'
    with open(audio_path + '/' + li + ".list") as f:
        for entry in f.readlines():
            if not '.' in entry:
                continue
            if entry.strip().split(':')[1] == "萝卜头":
                print entry
