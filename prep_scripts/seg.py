# import the necessary packages

from __future__ import division
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import argparse
import cv2
import h5py
import numpy as np
import multiprocessing as mp
import traceback, sys
from collections import Counter
from skimage.segmentation import mark_boundaries
import sys
import glob

img_folder= 'img/'
output_file= 'seg.h5'
list=glob.glob('img_bg/*jpg')
idx =0
# img=sys.argv[1]
for x in (list):
	image = cv2.imread(x)
	segments = slic(img_as_float(image), n_segments = 30 , sigma = 5)
	areas = []
	labels=[]
	s= segments.reshape(segments.shape[1]*segments.shape[0])
	word_count = Counter(s)
	occ=word_count.items()
	for i in occ:
		areas.append(i[1])
		labels.append(i[0]+1)

	# print (np.array(areas))
	# print (np.array(labels))

# output h5 file:
	dbo = h5py.File(output_file,'a')
	mask_dset = dbo.create_dataset(x.split('/')[1], data=segments)
	mask_dset.attrs['area'] = areas
	mask_dset.attrs['label'] = labels
	# print(x.split('/')[1])
	idx +=1
	print(idx)
	# print(areas)
# 	fig = plt.figure("Superpixels -- %d segments" % (1))
# 	ax = fig.add_subplot(1, 1, 1)
# 	ax.imshow(mark_boundaries(image, segments))
# 	ax = fig.add_subplot(1, 1, 1)
# 	ax.imshow(mark_boundaries(image, segments))
# 	plt.show()
dbo.close()
