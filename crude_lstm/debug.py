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
learning_rate = 0.0005
training_iters = 10000000000
batch_size = 1024
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


#for v in tf.all_variables():
#    print (v.name)
# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()
#load data

train_true_data = np.load(data_path + "train_true_data.npy")
train_fake_data = np.load(data_path + "train_fake_data.npy")
eva_true_data = np.load(data_path + "eva_true_data.npy")
eva_fake_data = np.load(data_path + "eva_fake_data.npy")

print ('train',train_true_data.shape)
print ('fake', train_fake_data.shape)

train_data = np.vstack((train_true_data, train_fake_data))
train_label = np.vstack((np.ones((train_true_data.shape[0], 1)), np.zeros((train_fake_data.shape[0], 1))))

eva_data = np.vstack((eva_true_data, eva_fake_data))
eva_label = np.vstack((np.ones((eva_true_data.shape[0], 1)), np.zeros((eva_fake_data.shape[0], 1))))

validate_best = 0


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(logdir, sess.graph)
    saver.restore(sess, "./checkpoint/extraoridinary-keep_prob-{}".format(KEEPPROB))
    y_pred = np.asarray(sess.run(pred, feed_dict={x:eva_data, K:1.0}), dtype=np.int16)
    from sklearn.metrics import confusion_matrix
    print (confusion_matrix(np.squeeze(eva_label), y_pred, labels=[0,1]))
    print (confusion_matrix(np.squeeze(eva_label), y_pred, labels=[1,0]))
