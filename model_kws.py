'''
KWS Model of LSTM+CTC
Robot.Ling@nationalchip.com
'''
import tensorflow as tf
#from tensorflow.models.rnn import rnn_cell, rnn
#from tensorflow.nn import rnn_cell, rnn
import numpy as np
import pdb

rnn_cell = tf.nn.rnn_cell
rnn = tf.nn.rnn


class KWSModel(object):
        def __init__(self,hyper_parameter):
                '''
                KWS rnn model, using ctc loss with lstm cells
                '''
                P = hyper_parameter
                #self.dropout = tf.placeholder(tf.float32)
                self.dropout = P.dropout
                self.batch_size = P.batch_size
                self.learning_rate = tf.Variable(
                        float(P.learning_rate), trainable=False)
                #self.learning_rate_decay_op = self.learning_rate.assign(
                #       self.learning_rate * P.lr_decay_factor)
                self.global_step = tf.Variable(0, trainable=False)
                self.dropout_keep_prob_lstm_input = self.dropout
                self.dropout_keep_prob_lstm_output = self.dropout

                #tf.scalar_summary("Learning Rate",self.learning_rate)
        def inference(self,wav_feats,input_lengths,P):
          with tf.name_scope("RNN_Inference"):
                num_steps = P.max_input_seq_length
                #Input feature extraction DNN
                w_i = tf.Variable(tf.truncated_normal(
                                [P.input_dim, P.hidden_size], stddev=0.2))
                b_i = tf.Variable(tf.constant(0., shape=[P.hidden_size]))

                rnn_inputs = [tf.nn.xw_plus_b(tf.squeeze(x,[1]),w_i,b_i) for x in
                                        tf.split(1,num_steps,wav_feats)]

                cell = rnn_cell.DropoutWrapper(
                        rnn_cell.BasicLSTMCell(P.hidden_size,state_is_tuple=True),
                        input_keep_prob=self.dropout_keep_prob_lstm_input,
                        output_keep_prob=self.dropout_keep_prob_lstm_output)

                cell =  rnn_cell.MultiRNNCell([cell] * P.num_layers,state_is_tuple=True)

                #set rnn init state to 0s
                initial_state = cell.zero_state(self.batch_size, tf.float32)

                rnn_outputs, state = tf.nn.dynamic_rnn(
                        cell,
                        tf.pack(rnn_inputs),
                        sequence_length=input_lengths,
                        initial_state=initial_state,
                        time_major=True
                        )
                w_o = tf.Variable(tf.truncated_normal(
                                [P.hidden_size, P.num_labels], stddev=0.2))
                b_o = tf.Variable(tf.constant(0., shape=[P.num_labels]))

                logits_ = [tf.nn.xw_plus_b(tf.squeeze(x,[0]),w_o,b_o) for x in
                                        tf.split(0,num_steps,rnn_outputs)]

                self.logits = tf.pack(logits_)

                self.W_i = w_i
                self.B_i = b_i
                self.W_o = w_o
                self.B_o = b_o

                #variable_summaries(w_o,"Weights")
                #variable_summaries(b_o,"Bias")

          return self.logits


        def ctc_greedy_decoder(self,logits,input_lengths,P):
          with tf.name_scope("CTC_greedy_decoder"):
                decoded, log_prob = tf.nn.ctc_greedy_decoder(
                        inputs = logits,
                        sequence_length = input_lengths,
                        merge_repeated=False )
                #Cast to same as target label
                decoded_sparse_tensor = tf.cast(decoded[0],tf.int32)
                #tf.scalar_summary("Predict Label",decoded_sparse_tensor.shape[1])
          return decoded_sparse_tensor,log_prob

        def ctc_beam_decoder(self,logits,input_lengths,P):
          with tf.name_scope("CTC_beam_decoder"):
                decoded,log_prob = tf.nn.ctc_beam_search_decoder(
                        inputs = logits,
                        sequence_length = input_lengths,
                        beam_width=100, top_paths=1, merge_repeated=False)
                decoded_sparse_tensor = tf.cast(decoded[0],tf.int32)
                #tf.scalar_summary("Predict beam Label",decoded_sparse_tensor.shape[1])
          return decoded_sparse_tensor,log_prob

        def loss(self,logits,sparse_labels,input_lengths,P):
          with tf.name_scope("CTC_Loss"):
                #input_seq_lengths = [P.max_input_seq_length]*P.batch_size
                input_seq_lengths = input_lengths
                #compute ctc loss
                self.ctc_loss = tf.nn.ctc_loss(
                        tf.pack(logits),
                        sparse_labels,
                        input_seq_lengths)
                self.mean_loss = tf.reduce_mean(self.ctc_loss)
                #tf.scalar_summary("CTC_loss",self.mean_loss)
          return self.mean_loss

        #Build training op
        def training(self,loss):
          with tf.name_scope("Train_op"):
                #self._train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
                #self._train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(loss)
                self._train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
                """
                self.params = tf.trainable_variables()
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                self.gradients = tf.gradients(loss, self.params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                        self.gradients, self.grad_clip)
                self._train_op = opt.apply_gradients(
                        zip(clipped_gradients, self.params),
                        global_step=self.global_step)
                #tf.scalar_summary("Gradients",self.gradients)
                """


          return self._train_op


class Inputs(object):
    def __init__(self,pool_size):
        self.pool_size = pool_size
        self._padding_len = 0
        self._wav = 0
        self._label = 0
        self._length = 0

    #def read_and_decode(self,filename_queue,input_dim,max_input_seq_length,label_dim):
    def batch_inputs(self,filename,batch_size,P):
      with tf.name_scope('inputs'):

        #File name queue for processing
        filename_queue = tf.train.string_input_producer(filename,num_epochs=P.max_epoch_num)

        #File Reader and feature extraction
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                serialized_example,
                features={
                        'length': tf.FixedLenFeature([],tf.int64),
                        'wav_feat': tf.FixedLenFeature([],tf.string),
                        'label': tf.VarLenFeature(tf.int64),
                        })

        length = tf.cast(features['length'], tf.int32)
        org_wav_feat   = tf.decode_raw(features['wav_feat'],tf.float32)
        self._sparse_label = tf.cast(features['label'],tf.int32)
        #Padding to max_input_seq_length
        self._pad_length = (P.max_input_seq_length - length) * P.input_dim
        self._padding = [[0,0],[0,self._pad_length]]
        self._wav_feat_padded = tf.pad([org_wav_feat],self._padding)

        #Reshape for feat and length
        self._wav_feat = tf.reshape(self._wav_feat_padded,[P.max_input_seq_length,P.input_dim])

        #Get Batch data in multi-thread
        return tf.train.shuffle_batch(
                [self._wav_feat,self._sparse_label,length],
                batch_size=batch_size, num_threads=2,
                capacity = self.pool_size + 3*batch_size,
                min_after_dequeue=self.pool_size
              )


