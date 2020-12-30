import h5py
import numpy as np
from PIL import Image
import os.path as osp
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import argparse
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import cPickle as cp
import glob
import re

im_dir = 'bg_img'
list=glob.glob('img/*jpg')
# img=sys.argv[1]
# seg_db = h5py.File('seg.h5','r')
# for x in (list):
#  seg = seg_db[x][:].astype('float32')
#  area = seg_db[x].attrs['area']
#  label = seg_db[x].attrs['label']
#  print(x)
#  print(label)
#  print(area)

 # pass

transformations = transforms.Compose([
    transforms.Resize(525),
    transforms.RandomCrop(512),

])
for idx, x in enumerate (list):
 img = Image.open(x)
 imgg = transformations(img)
 # name = x[1].split('/')[1]
 imgg.save('img_bg/img' + str(idx) + '.jpg')
