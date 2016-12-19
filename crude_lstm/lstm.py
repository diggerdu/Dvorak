from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pickle
import time

data_path = "../data/"
logdir = './logs'
# Parameters
learning_rate = 0.0001
training_iters = 10000000000
batch_size = 1
display_step = 12
milestone = 0.78

# Network Parameters
n_input = 26   #26 dim mfcc feature
n_steps = 60   #the length of input sequence 
n_hidden = 64     # the number of hidden unit
n_classes = 1   # oral / no oral

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

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

    # Linear activation, using rnn inner loop last output
    return tf.sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])


pred = RNN(x, weights, biases)

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
    writer = tf.train.SummaryWriter(logdir, sess.graph)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        idx = np.random.choice(train_data.shape[0], batch_size)
        batch_x = train_data[idx]
        batch_y = train_label[idx]
        embark = time.time()
        _, loss_sum_ = sess.run([optimizer, loss_sum], feed_dict={x: batch_x, y: batch_y})
        writer.add_summary(loss_sum_, global_step = step)
        print (time.time() - embark,"s per batch")
        if step % display_step == 0:
            accurancy_sum_, accurancy_ = sess.run([accurancy_sum, accurancy], feed_dict={x: eva_data, y: eva_label})
            writer.add_summary(accurancy_sum_, global_step = step)
            print ("after %d epoch, thus far training accurancy is : %f" %(step, accurancy_))
            if  accurancy_ > milestone:
                saver.save(sess, "./checkpoint/model.ckpt") 
                accurancy_ = sess.run(accurancy, feed_dict={x: eva_data, y: eva_label})
                
                print ("####thus far validate accurancy is : %f #####" %(accurancy_))
                if accurancy_ > validate_best:
                    validate_best = accurancy_
                    if validate_best > 0.86:
                        saver.save(sess, "./checkpoint/extraoridinary.ckpt")
                print ("####thus far best validate accurancy is : %f #####" %(validate_best))
                np.save("real.npy", eva_label)
        step += 1
    print("Optimization Finished!")
