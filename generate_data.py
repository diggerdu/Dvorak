#!/usr/bin/env python 
# coding=utf-8
import os
import pickle
import string
import mfcc
import numpy as np
import scipy.io.wavfile as wav
from pypinyin import pinyin
import matplotlib.pyplot as plt
from collections import Counter
from random import randint


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

    
trunc_len = 60
amp_thres = 2000
eva_audio_path = "audio/eva/"
train_audio_path = "audio/train/"
eva_file_dict = pickle.load(open("./audio/eva_dict", "rb"))
train_file_dict = pickle.load(open("./audio/train_dict", "rb"))

output_path = "data/"
w_len = 0.032
w_step = 0.016
filter_num = 26
n_classes = 26

eva_data = list()
eva_label = list()
eva_label_len = list()

train_data = list()
train_label = list()
train_label_len = list()

#generate encode & decode dict
decode_dict = dict(enumerate(string.lowercase))
encode_dict = {v:k for k,v in decode_dict.items()}



max_label_len = 0
for file_name, label in eva_file_dict.items():
    (rate, audio_ori) = wav.read(eva_audio_path + file_name)
    for pre_idx in xrange(audio_ori.shape[0]):
        if audio_ori[pre_idx] > amp_thres:
            break
    for sur_idx in xrange(audio_ori.shape[0] - 1, 0 , -1):
        if audio_ori[sur_idx] > amp_thres:
            break
    audio_ori = audio_ori[pre_idx:sur_idx]
    tmp, _ = mfcc.fbank(audio_ori, samplerate = rate, win_length = w_len,\
                win_step = w_step)
    #print tmp.shape[0]
    if tmp.shape[0] > trunc_len:
        continue
    else:
        pre_pad = randint(0, trunc_len - tmp.shape[0])
        sur_pad = trunc_len - tmp.shape[0] - pre_pad
        tmp = np.concatenate((np.zeros((pre_pad, filter_num)), tmp, np.zeros((sur_pad, filter_num))), axis = 0)
        eva_data.append(tmp)

    label = ''.join([str(zi[0]) for zi in pinyin(unicode(label, 'utf-8'), style=0, \
            heteronym=True)])

    eva_label_len.append(len(label))
    label = map(lambda c:encode_dict[c], list(label))

    if len(label) > max_label_len:
        max_label_len = len(label)
    eva_label.append(label)

for l in eva_label:
    for i in range(max_label_len - len(l)):
        l.append(-1)
print "########",np.asarray(eva_label).shape

eva_data = np.asarray(eva_data)
np.save(output_path + 'eva_data', eva_data)
np.save(output_path + 'eva_label', np.asarray(eva_label))

np.save(output_path + 'eva_label_len', np.asarray(eva_label_len))
del eva_data, eva_label, eva_label_len


max_label_len = 0
for file_name, label in train_file_dict.items():
    (rate, audio_ori) = wav.read(train_audio_path + file_name)
    for pre_idx in xrange(audio_ori.shape[0]):
        if audio_ori[pre_idx] > amp_thres:
            break
    for sur_idx in xrange(audio_ori.shape[0] - 1, 0 , -1):
        if audio_ori[sur_idx] > amp_thres:
            break
    audio_ori = audio_ori[pre_idx:sur_idx]
    tmp, _ = mfcc.fbank(audio_ori, samplerate = rate, win_length = w_len,\
                win_step = w_step)
    if tmp.shape[0] > trunc_len:
        continue
    else:
        pre_pad = randint(0, trunc_len - tmp.shape[0])
        sur_pad = trunc_len - tmp.shape[0] - pre_pad
        tmp = np.concatenate((np.zeros((pre_pad, filter_num)), tmp, \
                np.zeros((sur_pad, filter_num))), axis = 0)
	train_data.append(tmp)
    label = ''.join([str(zi[0]) for zi in pinyin(unicode(label, 'utf-8'), style=0, \
            heteronym=True)])
    label = map(lambda c:encode_dict[c], list(label))
    train_label_len.append(len(label))
    if len(label) > max_label_len:
        max_label_len = len(label)
    train_label.append(label)

for l in train_label:
    for i in range(max_label_len - len(l)):
        l.append(-1)


train_data = np.asarray(train_data)
np.save(output_path + 'train_data', train_data)
np.save(output_path + 'train_label', np.asarray(train_label))
np.save(output_path + 'train_label_len', np.asarray(train_label_len))
del train_data, train_label, train_label_len


