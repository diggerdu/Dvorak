from __future__ import print_function
from __future__ import division
import tensorflow as tf
from ultis import sparse_tuple_from
import numpy as np
import pickle
import time
from six.moves import xrange as range


data_path = "../data/"
# Parameters
learning_rate = 0.0001
training_iters = 10000000000
batch_size = 512
display_step = 100
milestone = 0.3

# Network Parameters
n_input = 26   #26 dim mfcc feature
n_steps = 60   #the length of input sequence 
n_hidden = 64     # the number of hidden unit
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

train_data = np.load(data_path + "train_data.npy")
train_label = np.load(data_path + "train_label.npy")
train_label_len = np.load(data_path + 'train_label_len.npy')
eva_data = np.load(data_path + "eva_data.npy")
eva_label = np.load(data_path + "eva_label.npy")
eva_label_len = np.load(data_path + 'eva_label_len.npy')


validate_best = 0
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        idx = np.random.choice(train_data.shape[0], batch_size)
        batch_x = train_data[idx]
        breakpoint = time.time()
        batch_targets = sparse_tuple_from(train_label[idx], n_classes)
        embark = time.time()
        _, ler_ = sess.run([optimizer, ler], feed_dict={x: batch_x, targets: batch_targets, \
            seq_len: np.repeat(n_steps, idx.shape[0])})
        if step % display_step == 0:
            if ler < milestone:
                saver.save(sess, "./checkpoint/model.ckpt")
        print ("{}s per batch and the label error rate is:{}".format(time.time()-embark, ler_))
        '''
        if step % display_step == 0:
            pred_data = sess.run(pred, feed_dict={x: eva_data[:50], y: eva_label[:50]})
            discrete_output = np.where(pred_data > 0.5, 1, 0)
            right = sum(discrete_output == batch_y) 
            print ("after %d epoch, thus far training accurancy is : %f" %(step, right / 50))
            if  right / batch_size > milestone:
                saver.save(sess, "./checkpoint/model.ckpt")
                pred_data = sess.run(pred, feed_dict={x: eva_data, y: eva_label})
                discrete_output = np.where(pred_data > 0.5, 1, 0)
                right = sum(discrete_output == eva_label)
                
                print ("####thus far validate accurancy is : %f #####" %(right / eva_data.shape[0]))
                if right / eva_data.shape[0] > validate_best:
                    validate_best = right / eva_data.shape[0]
                    if validate_best > 0.86:
                        saver.save(sess, "./checkpoint/extraoridinary.ckpt")
                print ("####thus far best validate accurancy is : %f #####" %(validate_best))
                wrong_item = np.argwhere((discrete_output == eva_label) == False)
                wrong_item = wrong_item[:,0]
                np.save("wrong_item.npy", wrong_item)
                np.save("prob.npy", pred_data)
                np.save("real.npy", eva_label)
        step += 1
    print("Optimization Finished!")
    '''
