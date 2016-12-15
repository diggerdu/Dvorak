from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
import pickle
import time
import os

INPUT_SIZE = 26
def save2txt(header_file, file_name, variable_name, *arrays):
    length = sum([arr.size for arr in arrays])
    np.savetxt(file_name, np.concatenate([arr.flatten() for arr in arrays]), delimiter=',', newline=',\n', comments='')
    declaration = "const double {0}[{1}]".format(variable_name, length)
    header_file.write("extern {};\n".format(declaration))
    tmp = open(file_name, "rb")
    content = tmp.read()
    tmp.close()
    tmp = open(file_name, "wb")
    tmp.write("{0} = {{\n".format(declaration))
    tmp.write(content)
    tmp.write("};\n")
    tmp.close()

def generate_src(*arrays):
    layer_num = int(len(arrays)/2)
    header_file = open("weights.h", "w+")
    source_file = open("weights.cpp", "w+")
    header_file.write("#include <stdio.h>\n")
    header_file.write("extern const size_t SIZE[{0}];\n".format(layer_num+1))
    source_file.write('#include "weights.h"\n')
    source_file.write("const size_t SIZE[{0}] = {{".format(layer_num+1))
    source_file.write("{},".format(INPUT_SIZE))
    for i in range(layer_num):
        source_file.write("{},".format(arrays[i*2+1].size))
    source_file.write("};\n")
    save2txt(header_file, "tmp_weights.txt", "WEIGHTS", 
            *[arrays[i*2] for i in range(layer_num)])
    save2txt(header_file, "tmp_biases.txt", "BIASES", 
            *[arrays[i*2+1] for i in range(layer_num)])
    header_file.close()
    tmp_weights = open("tmp_weights.txt", "r+")
    tmp_biases = open("tmp_biases.txt", "r+")
    source_file.write(tmp_weights.read())
    source_file.write(tmp_biases.read())
    source_file.close()
    tmp_weights.close()
    tmp_biases.close()
    os.remove("tmp_weights.txt")
    os.remove("tmp_biases.txt")


data_path = "../data/"
checkpoint_path = "checkpoint/model.ckpt"

saver = tf.train.import_meta_graph("checkpoint/model.ckpt.meta")
get_data_op = list()
for var in tf.trainable_variables():
    get_data_op.append(var)

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    weights = sess.run(get_data_op)
    weights[-2] = np.append(weights[-2], (0))
    for i in range(len(weights)):
        weights[i] = np.transpose(weights[i])
        print (weights[i].shape)
    print (weights[0][0].shape, weights[0][0])
    np.savetxt("debug.txt",weights[0].flatten())
    generate_src(*weights)

