from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import mfcc


# Network Parameters
n_input = 26   #26 dim mfcc feature
n_steps = 60   #the length of input sequence 
n_hidden = 128     # the number of hidden unit
n_classes = 1   # oral / no oral

#audio coefficient
w_len = 0.032
w_step = 0.032
sample_rate = 8000


class Predictor():
	def __init__(self):
		saver = tf.train.import_meta_graph('./backup/extraoridinary.ckpt.meta')
		self.sess = tf.Session()
		saver.restore(self.sess, "./backup/extraoridinary.ckpt")
		self.pred = self.sess.graph.get_operation_by_name('dropout/mul').outputs[0]
                self.x = self.sess.graph.get_operation_by_name('Placeholder').outputs[0]
		self.K = self.sess.graph.get_operation_by_name('Placeholder_2').outputs[0]
	def predict(self, audio):
		mfcc_feature,_ = mfcc.fbank(np.asarray(audio), samplerate = sample_rate, win_length = w_len,\
                win_step = w_step)
		assert (mfcc_feature.shape[0] == 60)
		mfcc_feature = np.expand_dims(mfcc_feature, axis = 0)
		label = self.sess.run(self.pred, feed_dict={self.x:mfcc_feature, self.K:1.0})
		print (label)
                print (label.shape)
                if label[0,0] > 0.5:
			return True
		else:
			return False
	def test(self):
		return 'I love heqinlin'
	
if __name__ == '__main__':
	p = Predictor()
	print ('model prepared')
        '''
        import librosa
        import os
        au_path = 'wrong_sample'
        for sam in os.listdir(au_path):
            print (p.predict(librosa.load(au_path+'/'+sam)[0][:15360]))
        '''
