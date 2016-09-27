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
output_path = "data_spectrum"
w_len = 0.032
w_step = 0.024
filter_num = 26

train_dict_file = open("audio/train_dict","rb")
eva_dict_file = open("audio/eva_dict","rb")
train_dict = pickle.load(train_dict_file)
eva_dict = pickle.load(eva_dict_file)
train_dict_file.close()
eva_dict_file.close()


def generate_data(file_path, file_label):
        (rate, audio_ori) = wav.read(file_path)
        for pre_idx in xrange(audio_ori.shape[0]):
            if audio_ori[pre_idx] > amp_thres:
                break
        for sur_idx in xrange(audio_ori.shape[0] - 1, 0 ,-1):
            if audio_ori[sur_idx] > amp_thres:
                break
        audio_ori = audio_ori[pre_idx : sur_idx]
        tmp = mfcc.wav_spectrum_power(audio_ori, samplerate=rate, win_length=w_len, win_step=w_step, filters_num=filter_num)
        if tmp.shape[0] > trunc_len:
            return None
        else:
            #half-open interval
            pre_pad = randint(0, trunc_len - tmp.shape[0])
            sur_pad = trunc_len -tmp.shape[0] - pre_pad
            tmp = np.concatenate((np.zeros((pre_pad, tmp.shape[1])), tmp, np.zeros((sur_pad, tmp.shape[1]))), axis = 0)
            tmp = tmp.reshape((1, tmp.shape[0], tmp.shape[1]))
            print tmp.shape
            return tmp
'''
os.chdir(output_path)
np.save("eva_fake.npy", eva_fake_data[1:])
np.save("eva_true.npy", eva_true_data[1:])
np.save("train_fake.npy", train_fake_data[1:])
np.save("train_true.npy", train_true_data[1:])

pickle.dump(eva_true_label, open("eva_true_label", "wb"))
pickle.dump(eva_fake_label, open("eva_fake_label", "wb"))
pickle.dump(train_true_label, open("train_true_label", "wb"))
pickle.dump(train_fake_label, open("train_fake_label", "wb"))

'''

if __name__ == '__main__':
    for key, value in train_dict.iteritems():
        print key
        tmp = generate_data("audio/train/" + key, value)
        if tmp is not None:
   
            print tmp.shape
        
