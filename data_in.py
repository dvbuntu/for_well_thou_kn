import os
import math
import numpy as np
import h5py
from tqdm import tqdm

dfile = 'font_data.h5'


ndata = 0
with h5py.File(dfile,'r') as h5f:
    label_list = list()
    # Get data size
    for k,v in h5f.items():
        ndata += v.shape[0]
        label_list.append(k)
    shape = (ndata, v.shape[1], v.shape[2])
    data = np.zeros(shape,dtype=np.uint8)
    # one-hot encoded
    labels = np.zeros((shape[0],len(h5f.values())),dtype=np.float16)
    # For every font
    for i,(k,v) in tqdm(enumerate(h5f.items()),total=len(label_list)):
        label = label_list.index(k)
        data[i*v.shape[0]:(i+1)*v.shape[0]]=v[:]
        labels[i*v.shape[0]:(i+1)*v.shape[0],label] = 1.



import matplotlib.pyplot as plt
plt.ion()

# plot images of a phrase in 5 fonts
num_phrases = len(data)//5
f, con = plt.subplots(5,1, sharex='col', sharey='row')
for i in range(5):
    con[i].pcolormesh(data[2000+i*num_phrases])

# split data into training/validation
np.random.seed(0)
ptrain_idx = np.random.choice(np.arange(num_phrases,dtype=np.uint32),replace=False,size=num_phrases//2)
ptest_idx = np.array([i for i in tqdm(np.arange(num_phrases)) if i not in train_idx],dtype=np.uint32)

train_idx = np.concatenate([ptrain_idx+num_phrases*i for i in range(5)])
test_idx = np.concatenate([ptest_idx+num_phrases*i for i in range(5)])


data_flat = data.reshape([-1,128*16])
train_flat = data_flat[train_idx]
test_flat = data_flat[test_idx]
train_labs = labels[train_idx]
test_labs = labels[test_idx]

# Build keras model
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import sklearn.metrics as metrics

model = Sequential()
model.add(Dense(16,input_shape=(128*16,)))
model.add(Activation('sigmoid'))
model.add(Dense(5))
model.add(Activation('softmax'))

sgd = SGD()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

h = model.fit(train_flat, train_labs, batch_size = 128, nb_epoch=2, validation_data = (test_flat,test_labs), verbose=1)

W1,b1,W2,b2 = model.get_weights()

sx, sy = (16,1)
f, con = plt.subplots(sx,sy, sharex='col', sharey='row')
con = con.reshape(sx,sy)
for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolormesh(W1[:,sy*xx+yy].reshape(16,128), cmap=plt.cm.hot) 

preds = np.argmax(model.predict(test_flat),axis=1)
labs = np.argmax(test_labs,axis=1)
conf = metrics.confusion_matrix(labs,preds)



