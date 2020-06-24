# Author: Ankush Gupta
# Date: 2015
"Script to generate font-models."

import pygame
from pygame import freetype
from text_utils import FontState
import numpy as np 
import matplotlib.pyplot as plt 
import pickle as cp


pygame.init()


ys = np.arange(8,200)
A = np.c_[ys,np.ones_like(ys)]
file1 = open("data/models/model.txt","w")

xs = []
models = {} #linear model

FS = FontState()
#plt.figure()
for i in range(len(FS.fonts)):
	print(i)
	font = freetype.Font(FS.fonts[i], size=12)
	h = []
	for y in ys:
		h.append(font.get_sized_glyph_height(int(y)))
	h = np.array(h)
	m,_,_,_ = np.linalg.lstsq(A,h)
	models[font.name] = m
	# np.savetxt('test.txt', m)
	# with open('data/models/model.txt', 'a') as outfile:
	# 	np.savetxt(outfile, m)
		# for i in m:
		# # file1.writelines(i + "\n")
		# 	np.savetxt(outfile, i)
	xs.append(h)
output = open('data.pkl', 'wb')
cp.dump(models, output)
output.close()
with open('font_px2pt.cp','wb') as f:
	cp.dump(models,f)
#plt.plot(xs,ys[i])
#plt.show()
