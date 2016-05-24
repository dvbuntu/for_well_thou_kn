import os
import math
import numpy as np
import h5py
from tqdm import tqdm
import keras

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

