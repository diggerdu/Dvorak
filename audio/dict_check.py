import os
import pickle


f = open("train_dict","rb")
dict = pickle.load(f)
print len(dict)
for key, value in dict.iteritems():
    dict[key] = value.strip()
f.close()
f = open("train_dict","wb")
pickle.dump(dict,f)
f.close()
