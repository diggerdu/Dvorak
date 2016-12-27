from __future__ import print_function
from __future__ import division
import tensorflow as tf
from ultis import sparse_tuple_from, dense_from_sparse, decode_from_arr
import numpy as np 
import pickle
import time
from six.moves import xrange as range


data_path = "../data/"
# Parameters
learning_rate = 0.00003
training_iters = np.inf
batch_size = 256
display_step = 1
milestone = 0.54

# Network Parameters
n_input = 26   #26 dim mfcc feature
n_steps = 60   #the length of input sequence 
n_hidden = 128     # the number of hidden unit
n_classes = 26 + 1   # oral / no oral

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
targets = tf.sparse_placeholder(tf.int32)
seq_len = tf.placeholder(tf.int32, [None])



def LSTM_CTC(x, seq_len):
    batch_s  = tf.shape(x)[0]
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 3, state_is_tuple=True)
    bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 3, state_is_tuple=True)
    #bi-directional rnn
    '''
    outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs=x,\
            sequence_length=seq_len, dtype=tf.float32, time_major=False)
    outputs = tf.concat(2, outputs) 
    
    '''
    #print ('#@$@$', outputs.get_shape().as_list())
    #uni-directional rnn
    outputs, _ = tf.nn.dynamic_rnn(fw_cell, x, dtype=tf.float32, time_major=False)
    
    ####transpose to [time, batch, output_size]
    outputs = tf.transpose(outputs, [1, 0, 2])
    outputs = tf.reshape(outputs, [-1, 1 * n_hidden])
    
    W = tf.Variable(tf.truncated_normal([1 * n_hidden, n_classes], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[n_classes]))
    
    logits = tf.matmul(outputs, W) + b
    logits = tf.reshape(logits, [n_steps, batch_s, n_classes])

    return logits
    

    
logits = LSTM_CTC(x, seq_len)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.ctc_loss(targets, logits, seq_len))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len, top_paths=3)

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
#label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()
#load data

eva_data = np.load(data_path + "eva_data.npy")
eva_label = np.load(data_path + "eva_label.npy")
eva_label_len = np.load(data_path + 'eva_label_len.npy')


validate_best = 0
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "./checkpoint/model.ckpt")
    eva_x = eva_data
    eva_targets = sparse_tuple_from(eva_label, n_classes)
    ler_, sparse_decoded= sess.run([ler, decoded], feed_dict={x:eva_x,\
            targets: eva_targets, seq_len: np.repeat(n_steps, eva_x.shape[0])})
    print ('label error rate in evaluate data set is: {}'.format(ler_))
    pred_dense_decoded = dense_from_sparse(sparse_decoded[0])
    truth_dense_decoded = dense_from_sparse(eva_targets)
    pred_decoded = decode_from_arr(pred_dense_decoded)
    print (pred_decoded[0])

    truth_decoded = decode_from_arr(truth_dense_decoded)
    pickle.dump(dict(zip(pred_decoded,truth_decoded)), open("./evaluate_result", "wb")) 
