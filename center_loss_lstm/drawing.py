from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time



data_path = "../data/"
log_path = "log"
# Parameters
learning_rate = 0.0003
training_iters = 10000000000
batch_size = 1024
display_step = 2
milestone = 0.97

# Network Parameters
n_input = 26   #26 dim mfcc feature
n_steps = 60   #the length of input sequence 
n_hidden = 64     # the number of hidden unit
n_classes = 1   # oral / no oral

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])


def RNN(x):

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
    
    weights_2 = tf.get_variable(name="weights_2", shape=[n_hidden, 2],\
            initializer=tf.truncated_normal_initializer())
    biases_2 = tf.get_variable(name="biases_2", shape=[2],\
            initializer=tf.truncated_normal_initializer())
    weights_1 = tf.get_variable(name="weights_1", shape=[2, 1],\
            initializer=tf.truncated_normal_initializer())
    biases_1 = tf.get_variable(name="biases_1", shape=[1],\
            initializer=tf.truncated_normal_initializer())
    
    
    drawing_layer = tf.sigmoid(tf.matmul(outputs[-1], weights_2) + biases_2)
    return tf.sigmoid(tf.matmul(drawing_layer, weights_1) + biases_1), drawing_layer


pred,drawing_layer= RNN(x)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
tf.scalar_summary("loss", cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()
summaries_op = tf.merge_all_summaries()
#load data

train_true_data = np.load(data_path + "train_true_data.npy")
train_fake_data = np.load(data_path + "train_fake_data.npy")
eva_true_data = np.load(data_path + "eva_true_data.npy")
eva_fake_data = np.load(data_path + "eva_fake_data.npy")


train_data = np.vstack((train_true_data, train_fake_data))
train_label = np.vstack((np.ones((train_true_data.shape[0], 1)), np.zeros((train_fake_data.shape[0], 1))))

eva_data = np.vstack((eva_true_data, eva_fake_data))
eva_label = np.vstack((np.ones((eva_true_data.shape[0], 1)), np.zeros((eva_fake_data.shape[0], 1))))

validate_best = 0
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "checkpoint/model.ckpt")
    points = sess.run(drawing_layer, feed_dict={x:train_data})
    print (points.shape)
    print (train_data.shape)
    x = points[:train_true_data.shape[0],0]
    y = points[:train_true_data.shape[0],1]
    print (x)
    for i in range(y.shape[0]):
        print (y[i])
    plt.scatter(x,y)
    #plt.scatter(points[:train_true_data.shape[0],0], points[:train_true_data.shape[0],1], c='b')
    #plt.scatter(points[train_true_data.shape[0]:,0], points[train_true_data.shape[0]:,1], c='r')
    plt.show()



