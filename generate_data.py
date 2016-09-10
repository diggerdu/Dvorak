#!/usr/bin/env python
# coding=utf-8
import os
import pickle
import mfcc
import numpy as np
import scipy.io.wavfile as wav
from random import randint
trunc_len = 60
amp_thres = 2000
audio_path = "audio"
output_path = "data_overlap"
suffix = ".list"
w_len = 0.032
w_step = 0.024
filter_num = 26

li_list = list()
for li in os.listdir(audio_path):
    if li.endswith(suffix):
        li_list.append(li.split('.')[0])

eva_true_data = np.zeros((1, trunc_len, filter_num))
eva_fake_data = np.zeros((1, trunc_len, filter_num))
train_true_data = np.zeros((1, trunc_len, filter_num))
train_fake_data = np.zeros((1, trunc_len, filter_num))

eva_true_label = list()
eva_fake_label = list()
train_true_label = list()
train_fake_label = list()

cnt1 = 0
cnt = 0
for li in li_list:
    cur_dir = audio_path + '/' + li.split('_')[0] + '/' + li.split('_')[1] + '/'
    with open(audio_path + '/' + li + ".list") as f:
        for entry in f.readlines():
            if not '.' in entry:
                continue
            wav_name = entry.strip().split(':')[0]
            label_name = entry.strip().split(':')[1]
            (rate, audio_ori) = wav.read(cur_dir + wav_name)
            for pre_idx in xrange(audio_ori.shape[0]):
                if audio_ori[pre_idx] > amp_thres:
                    break
            for sur_idx in xrange(audio_ori.shape[0] - 1, 0 ,-1):
                if audio_ori[sur_idx] > amp_thres:
                    break
            audio_ori = audio_ori[pre_idx : sur_idx]
            tmp, _ = mfcc.fbank(audio_ori, samplerate = rate, win_length = w_len,\
                            win_step = w_step        
                    )
            if tmp.shape[0] > trunc_len:
                continue
            else:
                #half-open interval
                pre_pad = randint(0, trunc_len - tmp.shape[0])
                sur_pad = trunc_len -tmp.shape[0] - pre_pad
                tmp = np.concatenate((np.zeros((pre_pad, filter_num)), tmp, np.zeros((sur_pad, filter_num))), axis = 0)
                tmp = tmp.reshape((1, tmp.shape[0], tmp.shape[1]))
                print tmp.shape
            if label_name == "萝卜头":
                if "test" in li:
                    eva_true_data = np.vstack((eva_true_data, tmp))
                    eva_true_label.append([0, entry, label_name])
                else:
                    train_true_data = np.vstack((train_true_data, tmp))
                    train_true_label.append([0, entry, label_name])
            else:
                if "test" in li:
                    eva_fake_data = np.vstack((eva_fake_data, tmp))
                    eva_fake_label.append([0, entry, label_name])
                else:
                    train_fake_data = np.vstack((train_fake_data, tmp))
                    train_fake_label.append([0, entry, label_name])
            
os.chdir(output_path)
np.save("eva_fake.npy", eva_fake_data[1:])
np.save("eva_true.npy", eva_true_data[1:])
np.save("train_fake.npy", train_fake_data[1:])
np.save("train_true.npy", train_true_data[1:])

pickle.dump(eva_true_label, open("eva_true_label", "wb"))
pickle.dump(eva_fake_label, open("eva_fake_label", "wb"))
pickle.dump(train_true_label, open("train_true_label", "wb"))
pickle.dump(train_fake_label, open("train_fake_label", "wb"))


