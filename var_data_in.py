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
    con[i].pcolormesh(data[1973+i*num_phrases])

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
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import sklearn.metrics as metrics

model = Sequential()
model.add(Dense(3,input_shape=(nrows*ncols,)))
model.add(Activation('sigmoid'))
model.add(Dense(5))
model.add(Activation('softmax'))

sgd = SGD()
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

h = model.fit(train_flat, train_labs, batch_size = 128, nb_epoch=4, validation_data = (test_flat,test_labs), verbose=1)

#W1,b1 = model.get_weights()
W1,b1,W2,b2 = model.get_weights()
num_param = 1024*7 + 7 + 7*5 + 5

sx, sy = (7,1)
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

### Build as a simple a model as possible
# Extract features, overall height and width
def get_dim(im):
    '''Given image, return nonzero width and height'''
    # Non-zero cols
    NC = np.where(np.max(im,axis=0) != 0)
    # overall width
    width = NC[0][-1] - NC[0][0]
    # Max of row
    NR = np.where(np.max(im,axis=1) != 0)
    # overall height
    height = NR[0][-1] - NR[0][0]
    return width,height

train_min = np.array([get_dim(d) for d in tqdm(data[train_idx])],dtype=np.uint8)
test_min = np.array([get_dim(d) for d in tqdm(data[test_idx])],dtype=np.uint8)

import sklearn.linear_model as lm
C = lm.LogisticRegression()
C.fit(train_min,np.argmax(train_labs,axis=1))
C.score(test_min,np.argmax(test_labs,axis=1))
preds2 = C.predict_proba(test_min)
# Tiny AIC
aic2 = 2* 15 - 2*metrics.log_loss(np.argmax(test_labs,axis=1),preds2)

# Three parameter model
def get_3(im):
    '''Given image, return nonzero width, bot, top'''
    # Non-zero cols
    NC = np.where(np.max(im,axis=0) != 0)
    # overall width
    width = NC[0][-1] - NC[0][0]
    # Max of row
    NR = np.where(np.max(im,axis=1) != 0)
    # overall height
    #height = NR[0][-1] - NR[0][0]
    return width,NR[0][0],NR[0][-1]

train_min3 = np.array([get_3(d) for d in tqdm(data[train_idx])],dtype=np.uint8)
test_min3 = np.array([get_3(d) for d in tqdm(data[test_idx])],dtype=np.uint8)

C3 = lm.LogisticRegression()
C3.fit(train_min3,np.argmax(train_labs,axis=1))
C3.score(test_min3,np.argmax(test_labs,axis=1))
preds3 = C3.predict_proba(test_min3)
# Tiny AIC
aic3 = 2* 20 - 2*metrics.log_loss(np.argmax(test_labs,axis=1),preds3)


