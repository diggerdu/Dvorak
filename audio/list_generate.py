import os
import pickle

train_list = "train.list"
eva_list = "eva.list"

train = open(train_list, "rb")
#eva = open(eva_list, "rb")

train_dict = dict()
eva_dict = dict()

for entry in train.readlines():
    file_name = entry.split(",")[0]
    label = entry.split(",")[1]
    file_name = file_name.split("/")[-1]
    train_dict.update({file_name:label})

pickle.dump(train_dict,open("train_dict","wb"))
train.close()
'''
for entry in eva.readlines():
    file_name = entry.split(",")[0]
    label = entry.split(",")[1]
    file_name = file_name.split("/")[-1]
    eva_dict.update({file_name:label})

pickle.dump(eva_dict,open("eva_dict","wb"))
eva.close()
'''

