import os
import gc

import sklearn
from sklearn.svm import SVC
from sklearn import cross_validation, grid_search
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
def train_svm_classifer(features, labels, model_output_path):
    
    params = [
        {
            'kernel':['linear'],
            'C':[1, 10, 100, 1000]
        },
        {
            'kernel':['rbf'],
            'C':[1, 10, 100, 1000],
            'gamma':[1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]

    svm = SVC(probability=True)
    clf = grid_search.GridSearchCV(svm, params, cv=32, n_jobs=15, verbose=3)
    clf.fit(features, labels)
    if os.path.exists(model_output_path):
        sklearn.externals.joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print ('Cannot save trained svm model to {0}'.format(model_output_path))
    return clf

if __name__ == '__main__':
    data_path = '../data/' 
    train_true_feature = np.load(data_path+'train_true_feature.npy')
    train_fake_feature = np.load(data_path+'train_fake_feature.npy')
    train_feature = np.vstack((train_true_feature, train_fake_feature))
    train_label = np.hstack((np.ones((train_true_feature.shape[0], )), np.zeros((train_fake_feature.shape[0], ))))
    del train_true_feature
    del train_fake_feature
    gc.collect()
    model = train_svm_classifer(train_feature, train_label, 'svm_model/best_model')
    eva_true_feature = np.load(data_path+'eva_true_feature.npy')
    eva_fake_feature = np.load(data_path+'eva_fake_feature.npy')
    eva_feature = np.vstack((eva_true_feature, eva_fake_feature))
    eva_predict = model.predict(eva_feature)
    eva_labels = np.hstack((np.ones(eva_true_feature.shape[0]), np.zeros(eva_fake_feature.shape[0])))
    print (confusion_matrix(eva_labels,eva_predict,labels=[0,1]))
    '''
    eva_true_feature = np.load(data_path+'eva_true_feature.npy')
    eva_fake_feature = np.load(data_path+'eva_fake_feature.npy')
    '''
