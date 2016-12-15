from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import pickle
import time

data_path = "../data/"
# Parameters
tower_name = "lstm_tower"
learning_rate = 0.0001
training_iters = 10000000000
batch_size = 1024
display_step = 50
milestone = 0.98
num_gpus = 2
# Network Parameters
n_input = 26   #26 dim mfcc feature
n_steps = 60   #the length of input sequence 
n_hidden = 64     # the number of hidden unit
n_classes = 1   # oral / no oral

train_true_data = np.load(data_path + "train_true_data.npy")
train_fake_data = np.load(data_path + "train_fake_data.npy")
eva_true_data = np.load(data_path + "eva_true_data.npy")
eva_fake_data = np.load(data_path + "eva_fake_data.npy")


train_data = np.vstack((train_true_data, train_fake_data))
train_label = np.vstack((np.ones((train_true_data.shape[0], 1)), np.zeros((train_fake_data.shape[0], 1))))

train_data = tf.cast(train_data, tf.float32)
trian_label = tf.cast(train_label, tf.float32)

eva_data = np.vstack((eva_true_data, eva_fake_data))
eva_label = np.vstack((np.ones((eva_true_data.shape[0], 1)), np.zeros((eva_fake_data.shape[0], 1))))



def RNN(batch):
    batch = tf.cast(batch, tf.float32)
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    batch = tf.transpose(batch, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    batch = tf.reshape(batch, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    batch = tf.split(0, n_steps, batch)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * 3, state_is_tuple=True)
    # Get lstm cell output
    outputs, states = rnn.rnn(cell, batch, dtype=tf.float32)
    
    out_weights = tf.get_variable(name="out_weights", shape=[n_hidden, n_classes],\
            initializer=tf.truncated_normal_initializer())
    out_biases = tf.get_variable(name="out_biases", shape=[n_classes], \
            initializer=tf.truncated_normal_initializer())
    # Linear activation, using rnn inner loop last output
    logits = tf.sigmoid(tf.matmul(outputs[-1], out_weights) + out_biases)
    return logits

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def tower_loss(scope, train_data, train_label, batch_size):
    idx = np.random.choice(train_data.get_shape().as_list()[0], batch_size)
    idx = tf.cast(idx, tf.int32)
    input_data = tf.gather(train_data, idx)
    labels = tf.cast(tf.gather(train_label, idx), tf.float32)
    logits = RNN(input_data)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels))    
    return loss, logits, labels
    

opt = tf.train.AdamOptimizer(learning_rate)
tower_grads = []
for i in xrange(num_gpus):
    with tf.device('/gpu:%d' % i):
        with tf.name_scope("%s_%d" % (tower_name, i)) as scope:
            loss, logits, labels = tower_loss(scope, train_data, train_label, batch_size)
            tf.get_variable_scope().reuse_variables()
            grads = opt.compute_gradients(loss)
            tower_grads.append(grads)
grads = average_gradients(tower_grads)
apply_gradient_op = opt.apply_gradients(grads)

init = tf.initialize_all_variables()
validate_best = 0

sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True
        ))
sess.run(init)
tf.train.start_queue_runners(sess=sess)
step = 0
while step*batch_size < training_iters:
    embark = time.time()
    _, loss_value , output_data, cur_labels= sess.run([apply_gradient_op, loss, logits, labels])
    print ((time.time() - embark) / 2, "s per batch")
    output_data = np.where(output_data > 0.5, 1, 0)
    right = sum(output_data == cur_labels)
    print (right / batch_size)
    print (loss_value)
    step = step + 1
