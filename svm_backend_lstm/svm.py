import numpy as np
from libsvm import svmutil

features_path = "features/" 
train_true = np.load(features_path+"train_true.npy").tolist()
train_fake = np.load(features_path+"train_fake.npy").tolist()
eva_true = np.load(features_path+"eva_true.npy").tolist()
eva_fake = np.load(features_path+"eva_fake.npy").tolist()

f = open("features.csv", "w+")
for li in train_true:
    f.write("+1 ")
    for idx, value in enumerate(li):
        f.write("{}:{} ".format(idx+1, value))
    f.write("\n")

for li in train_fake:
    f.write("-1 ")
    for idx, value in enumerate(li):
        f.write("{}:{} ".format(idx+1, value))
    f.write("\n")

f.close()


train_data = train_true[::]
train_data.extend(train_fake)
tmp = list()
for li in train_data:
    tmp.append(dict(enumerate(li)))
train_data = tmp
train_label = [1 for i in range(len(train_true))] + [-1 for i in range(len(train_fake))] 

eva_data = eva_true[::]
eva_data.extend(eva_fake)
tmp = list()
for li in eva_data:
    tmp.append(dict(enumerate(li)))
eva_data = tmp
eva_label = [1 for i in range(len(eva_true))] + [-1 for i in range(len(eva_fake))]

model = svmutil.svm_train([1 for i in range(len(train_true))] + [-1 for i in range(len(train_fake))], train_data, '-c 0.03125 -g 0.25')
print type(model)
p_label, p_acc, p_val = svmutil.svm_predict([1 for i in range(len(eva_true))] + [-1 for i in range(len(eva_fake))], eva_data, model)

print p_acc



