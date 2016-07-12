import os
import math
import numpy as np
import h5py
from tqdm import tqdm

dfile = 'var_font_data.h5'

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
    con[i].pcolormesh(data[1973+i*num_phrases],cmap=plt.cm.hot_r)

# split data into training/validation
# do ahead of time
np.random.seed(0)
ptrain_idx = np.random.choice(np.arange(num_phrases,dtype=np.uint32),replace=False,size=num_phrases//2)
ptest_idx = np.array([i for i in tqdm(np.arange(num_phrases)) if i not in ptrain_idx],dtype=np.uint32)

train_idx = np.concatenate([ptrain_idx+num_phrases*i for i in range(5)])
test_idx = np.concatenate([ptest_idx+num_phrases*i for i in range(5)])

nrows = 16
ncols = 64
data_flat = data.reshape([-1,nrows*ncols])
train_flat = data_flat[train_idx]
test_flat = data_flat[test_idx]
train_labs = labels[train_idx]
test_labs = labels[test_idx]

# Build keras model
import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Embedding, Input, merge
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import sklearn.metrics as metrics

model = Sequential()
model.add(Dense(4,input_shape=(nrows*ncols,)))
model.add(Activation('sigmoid'))
model.add(Dense(5))
model.add(Activation('softmax'))

sgd = SGD()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

h = model.fit(train_flat, train_labs, batch_size = 32, nb_epoch=4, validation_data = (test_flat,test_labs), verbose=1)

#W1,b1 = model.get_weights()
W1,b1,W2,b2 = model.get_weights()
num_param = 1024*4 + 4 + 4*5 + 5

sx, sy = (4,1)
f, con = plt.subplots(sx,sy, sharex='col', sharey='row')
con = con.reshape(sx,sy)
for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolormesh(W1[:,sy*xx+yy].reshape(nrows,ncols), cmap=plt.cm.hot) 

preds = np.argmax(model.predict(test_flat),axis=1)
labs = np.argmax(test_labs,axis=1)
conf = metrics.confusion_matrix(labs,preds)
predsp = model.predict_proba(test_flat)
aic = 2* num_param - 2*metrics.log_loss(np.argmax(test_labs,axis=1),predsp)

# test blocking up data
num_blocks = 8
d = data[2000:2002]
B = np.swapaxes(d,1,2).reshape((2,8,8,16))
blocked_d = np.swapaxes(B,2,3)
f, con = plt.subplots(2,num_blocks, sharex='col', sharey='row')
for j in range(2):
    for i in range(num_blocks):
        con[j,i].pcolormesh(blocked_d[j,i],cmap=plt.cm.hot_r)

# reshape input into 8 blocks of 16x8
data_block = data.swapaxes(1,2).reshape((-1,8,8,16)).swapaxes(2,3)
data_block_flat = data_block.reshape((-1,8,8*16))
train_block = data_block[train_idx]
test_block = data_block[test_idx]
train_block_flat = data_block_flat[train_idx]
test_block_flat = data_block_flat[test_idx]
train_block_swap = train_block.swapaxes(0,1).reshape((8,-1,1,16,8))
test_block_swap = test_block.swapaxes(0,1).reshape((8,-1,1,16,8))

# RNN model
# need 8 inputs
blocks = [Input(shape=(1,nrows,8), name='main_input_{0}'.format(b)) for b in range(num_blocks)]
# convolution for each input...but I want weights to be the same
conv = Convolution2D(8,3,3,border_mode='same',
        activation='relu')
convs = [conv(b) for b in blocks]
pools = [MaxPooling2D(pool_size=(2,2))(c)
        for c in convs]
conv2 = Convolution2D(8,3,3,border_mode='same',
        activation='relu')
convs2 = [conv2(p) for p in pools]
pools2 = [MaxPooling2D(pool_size=(2,2))(c)
         for c in convs2]
flats = [Flatten()(p) for p in pools2]
M = merge(flats, mode='concat', concat_axis=1)
R = Reshape((8,64))(M)
L = LSTM(output_dim=16, activation='sigmoid', inner_activation='hard_sigmoid')(R)
D = Dropout(0.5)(L)
O = Dense(5, activation='sigmoid')(D)
model = Model(input=blocks, output=[O])

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

n = 1000000
h = model.fit(list(train_block_swap[:,:n]), train_labs[:n], batch_size = 32, nb_epoch=10, validation_data = (list(test_block_swap[:,:n]),test_labs[:n]), verbose=1)

# Get the weights
W = model.get_weights()
np.savez_compressed('cnnrnn.npz',W)
num_param = 20877

# Look at the convolutional weight inputs
sx, sy = (8,1)
f, con = plt.subplots(sx,sy, sharex='col', sharey='row')
con = con.reshape(sx,sy)
for xx in range(sx):
    for yy in range(sy):
        con[xx,yy].pcolormesh(W[0][xx].reshape(3,3), cmap=plt.cm.gray) 

# confusion matrix
preds = np.argmax(model.predict(list(test_block_swap)),axis=1)
labs = np.argmax(test_labs,axis=1)
conf = metrics.confusion_matrix(labs,preds)

