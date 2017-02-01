from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pickle
import time

data_path = "../data/"
logdir = './logs'
KEEPPROB = 0.7
# Parameters
learning_rate = 0.0002
training_iters = 10000000000
batch_size = 1596
display_step = 2
milestone = 0.78

# Network Parameters
n_input = 26   #26 dim mfcc feature
n_steps = 60   #the length of input sequence 
n_hidden = 128     # the number of hidden unit
n_classes = 1   # oral / no oral

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
K = tf.placeholder(tf.float32)
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, keep_prob, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * 3, state_is_tuple=True)
    # Get lstm cell output
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)
    outputs_drop = tf.nn.dropout(outputs[-1], keep_prob)
    # Linear activation, using rnn inner loop last output
    return tf.sigmoid(tf.matmul(outputs_drop, weights['out']) + biases['out'])


pred = RNN(x, K, weights, biases)

#measure accurancy
accurancy = tf.reduce_mean(tf.cast(tf.equal(tf.round(pred), y), tf.float32))

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#summary operation
loss_sum = tf.summary.scalar('loss', cost)
accurancy_sum = tf.summary.scalar('accurancy', accurancy)
#summary_writer = tf.train.SummaryWriter(logdir, sess)


# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()
#load data

train_true_data = np.load(data_path + "train_true_data.npy") / 180000.00
train_fake_data = np.load(data_path + "train_fake_data.npy") / 180000.00
eva_true_data = np.load(data_path + "eva_true_data.npy") / 180000.00
eva_fake_data = np.load(data_path + "eva_fake_data.npy") / 180000.00


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "./checkpoint/extraoridinary-keep_prob-{}".format(KEEPPROB))
    feature_op = sess.graph.get_operation_by_name('dropout/mul').outputs[0]
    '''
    train_true_feature = sess.run(feature_op, feed_dict={x:train_true_data, K:1.0}) 
    train_fake_feature = sess.run(feature_op, feed_dict={x:train_fake_data, K:1.0}) 
    eva_true_feature = sess.run(feature_op, feed_dict={x:eva_true_data, K:1.0}) 
    eva_fake_feature = sess.run(feature_op, feed_dict={x:eva_fake_data, K:1.0}) 
    
    
    
    np.save(data_path + 'train_true_feature', train_true_feature)
    np.save(data_path + 'train_fake_feature', train_fake_feature)
    np.save(data_path + 'eva_true_feature', eva_true_feature)
    np.save(data_path + 'eva_fake_feature', eva_fake_feature)
    '''
    import mfcc
    import os
    import librosa
    au_path = 'wrong_sample'
    input_data = list()
    for sam in os.listdir(au_path):
        tmp, sr = librosa.load(au_path+'/'+sam, sr=8000)
        assert sr==8000
        tmp = tmp[:15360]
        mfcc_f,_ = mfcc.fbank(tmp, samplerate=sr,win_length=0.032,win_step=0.032)
        print (mfcc_f.shape)
        input_data.append(mfcc_f)
    input_data = np.asarray(input_data)
    print (input_data.shape)
    assert(input_data.shape[1] == 60)
    wa_feature = sess.run(feature_op, feed_dict={x:input_data, K:1.0})
    print (wa_feature.shape)
    np.save('wa_feature', wa_feature)
