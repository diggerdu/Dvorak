from __future__ import print_function
from __future__ import division
import tensorflow as tf
import sklearn.externals.joblib as joblib
from sklearn.svm import SVC
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import mfcc
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Network Parameters
n_input = 26   #26 dim mfcc feature
n_steps = 60   #the length of input sequence 
n_hidden = 128     # the number of hidden unit
n_classes = 1   # oral / no oral
KEEPPROB = 0.7
#audio coefficient
w_len = 0.032
w_step = 0.032
sample_rate = 8000

#path
svm_model_path = 'svm_model/best_model'



class Predictor():
	def __init__(self):
                saver = tf.train.import_meta_graph('./checkpoint/extraoridinary-keep_prob-0.7.meta')
                self.sess = tf.Session()
		saver.restore(self.sess, "./checkpoint/extraoridinary-keep_prob-0.7")
		self.feature = self.sess.graph.get_operation_by_name('dropout/mul').outputs[0]
                self.x = self.sess.graph.get_operation_by_name('Placeholder').outputs[0]
		self.K = self.sess.graph.get_operation_by_name('Placeholder_2').outputs[0]
                self.svm_model = joblib.load(svm_model_path)

	def predict(self, audio):
		mfcc_feature,_ = mfcc.fbank(np.asarray(audio), samplerate = sample_rate, win_length = w_len,\
                win_step = w_step)
		assert (mfcc_feature.shape[0] == n_steps)
		mfcc_feature = np.expand_dims(mfcc_feature, axis = 0)
		lstm_feature = self.sess.run(self.feature, feed_dict={self.x:mfcc_feature, self.K:1.0})
                probability = self.svm_model.predict(lstm_feature)
                return probability
if __name__ == '__main__':
	p = Predictor()
	print ('model prepared')
	import librosa
        import os
        au_path = 'wrong_sample'
        for sam in os.listdir(au_path):
            print (p.predict(librosa.load(au_path+'/'+sam, sr=8000)[0][:15360]))
