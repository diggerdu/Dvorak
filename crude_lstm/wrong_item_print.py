import pickle
import numpy as np

file_list = pickle.load(open("../data/eva_true_label","rb"))
file_list.extend(pickle.load(open("../data/eva_fake_label", "rb")))
#print file_list
file_dict = pickle.load(open("../audio/eva_dict","rb"))


output_file = open("oops.list", "wb")

wrong_file_list = list()
wrong_label_list = list()
wrong_item = np.load("wrong_item.npy")
for Idx in wrong_item:
    wrong_file_list.append(file_list[Idx])
    wrong_label_list.append(file_dict.get(file_list[Idx]))
    output_file.writelines("{0},{1}{2}".format(wrong_file_list[-1], wrong_label_list[-1], '\n'))
    print "../audio/eva/{0}".format(wrong_file_list.pop())
output_file.close()
