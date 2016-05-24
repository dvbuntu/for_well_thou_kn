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
num_chars = 16
with open(bill,'r') as f:
    phrases = [l.strip()[:num_chars] for l in f if len(l.strip()) > num_chars]


# squish 16 characters into 128 pixels wide
font_size = 12
img_size = (128,16)

ndata = len(fonts) * len(phrases)

with h5py.File('font_data.h5','w') as h5f:
    for f in fonts:
        font = ImageFont.truetype(f,font_size)
        base = os.path.basename(f)
        dset = h5f.create_dataset(base, (len(phrases),img_size[1],img_size[0]),dtype='uint8')
        for i,p in tqdm(enumerate(phrases),total=len(phrases)):
            w, h = font.getsize(p)
            # some fonts are weird with big offsets
            ow, oh = font.getoffset(p)
            w = w+ow
            h = h+oh
            img = Image.new('L', img_size)
            draw = ImageDraw.Draw(img)
            draw.text(((img_size[0]-w)/2,(img_size[1]-h)/2), p, fill="white", font=font)
            b = np.flipud(img)
            dset[i] = b

