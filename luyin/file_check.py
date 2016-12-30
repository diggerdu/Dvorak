import os
import pickle

audio_dir = 'audio'
lab_dir = 'lab'

audio_list = os.listdir(audio_dir)

file_dict = dict()
for f in os.listdir(lab_dir):
    if f[:-4] not in audio_list:
        print 'opps',f
        continue
    file_dict.update({f[:-4]:open(lab_dir+ '/' + f).read()})

pickle.dump(file_dict,open('file_dict','wb'))


