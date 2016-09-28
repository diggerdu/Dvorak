from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pickle


data_path = "../data_spectrum/"
# Parameters
learning_rate = 0.00001
training_iters = 10000000000
batch_size = 1024
display_step = 2
milestone = 0.84

# Network Parameters
n_input = 257   # MNIST data input (img shape: 28*28)
n_steps = 60
n_hidden = 64     # hidden layer num of features
n_classes = 1   # oral / no oral

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'filters' : tf.Variable(tf.random_normal([ 257, 26])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    print (weights['filters'].get_shape())
    x_shape = x.get_shape().as_list()
    print (x_shape, type(x_shape), type(x_shape[0]))
    x = tf.reshape(x, [ -1, x_shape[2]])
    print (x.get_shape())
    x = tf.matmul(x, weights['filters'])
    print (x.get_shape())
    x = tf.reshape(x, [-1, x_shape[1], 26])
    print (x.get_shape())
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, 26])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * 3, state_is_tuple=True)
    # Get lstm cell output
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)


    # Linear activation, using rnn inner loop last output
    return tf.sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])


pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

#load data

posi_train_data = np.load(data_path + "posi_train.npy")
nega_train_data = np.load(data_path + "nega_train.npy")
posi_eva_data = np.load(data_path + "posi_eva.npy")
nega_eva_data = np.load(data_path + "nega_eva.npy")

train_data = np.vstack((posi_train_data, nega_train_data))
train_label = np.vstack((np.ones((posi_train_data.shape[0], 1)), np.zeros((nega_train_data.shape[0], 1))))
print (train_data.shape)

eva_data = np.vstack((posi_eva_data, nega_eva_data))
eva_label = np.vstack((np.ones((posi_eva_data.shape[0], 1)), np.zeros((nega_eva_data.shape[0], 1))))

validate_best = 0
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        idx = np.random.choice(train_data.shape[0], batch_size)
        batch_x = train_data[idx]
        batch_y = train_label[idx]
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:    
            pred_data = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
            discrete_output = np.where(pred_data > 0.5, 1, 0)
            right = sum(discrete_output == batch_y)
            print ("after %d epoch, thus training accurancy is : %f" %(step, right / batch_size))
            if  right / batch_size > milestone:
                pred_data = sess.run(pred, feed_dict={x: eva_data, y: eva_label})
                discrete_output = np.where(pred_data > 0.5, 1, 0)
                right = sum(discrete_output == eva_label)
                print ("####thus validate accurancy is : %f #####" %(right / eva_data.shape[0])) 
                if right / eva_data.shape[0] > validate_best:
                    validate_best = right / eva_data.shape[0]
                print ("####thus best validate accurancy is : %f #####" %(validate_best)) 
                np.save("prob.npy", pred_data)
                np.save("real.npy", eva_label)
        step += 1
    print("Optimization Finished!")
