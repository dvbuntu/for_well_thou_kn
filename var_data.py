from PIL import ImageFont, ImageDraw, Image
import glob
import os
import math
import numpy as np
import h5py
from tqdm import tqdm

# 5 interesting fonts
fonts = glob.glob('fonts/*.otf')

# 100000 lines of shakespeare
bill = 't8.shakespeare.txt'
max_chars = 32
with open(bill,'r') as f:
    phrases = [l.strip()[:max_chars] for l in f if len(l.strip()) > max_chars]


# squish characters into 64 pixels wide
base_size = 16
img_size = (64,16)

ndata = len(fonts) * len(phrases)

# how many pixels we might jitter (+ or -, x and y)
jspan = 3

with h5py.File('var_font_data.h5','w') as h5f:
    for f in fonts:
        font = ImageFont.truetype(f,base_size)
        base = os.path.basename(f)
        dset = h5f.create_dataset(base, (len(phrases),img_size[1],img_size[0]),dtype='uint8')
        for i,p in tqdm(enumerate(phrases),total=len(phrases)):
            h = 0
            font_size = base_size
            while h < img_size[1]-1:
                font = ImageFont.truetype(f,font_size)
                w, h = font.getsize(p)
                font_size +=1 
            # some fonts are weird with big offsets
            ow, oh = font.getoffset(p)
            w = w+ow
            h = h+oh
            # Randomly jitter
            jx = np.random.randint(-1*jspan,jspan)
            jy = np.random.randint(-1*jspan,jspan)
            img = Image.new('L', img_size)
            draw = ImageDraw.Draw(img)
            draw.text(((img_size[0]-w)/2+jx,
                       (img_size[1]-h)/2+jy),
                       p, fill="white", font=font)
            b = np.flipud(img)
            dset[i] = b

