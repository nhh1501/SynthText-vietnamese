# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
import h5py 
from common import *
import re

import imageio
def cre_output(imname,img, senBB, txt):
    # ts = str(int(time() * 1000))
    imname = re.sub('.jpg', '', imname)
    imname = re.sub('.png', '', imname)
    # executed only if --output-masks flag is set
    prefix = "img/" + "gt_" + imname

    # imageio.imwrite(prefix + "_original.png", img)
    imageio.imwrite("img/" + imname + '.jpg', img)
    #
    # merge masks together:
    # merged = reduce(lambda a, b: np.add(a, b), res[0]['masks'])
    # since we just added values of pixels, need to bring it back to 0..255 range.
    # merged = np.divide(merged, len(res[0]['masks']))
    # imageio.imwrite(prefix + "_mask.png", merged)

    # print bounding boxes
    f = open(prefix + ".txt", "w+", encoding='utf8')
    # f = codecs.open(prefix + "_bb.txt", "w+","utf8")
    # bbs = res[0]['senBB']
    bbs = senBB
    boxes = np.swapaxes(bbs, 2, 0)
    # words = re.sub(' +', ' ', ' '.join(res[0]['txt']).replace("\n", " ")).strip().split(" ")
    # words = res[0]['txt']
    words = txt
    assert len(boxes) == len(words)
    for j in range(len(boxes)):
        as_strings = np.char.mod('%f', boxes[j].flatten())
        f.write(",".join(as_strings) + "," + words[j] + "\n")
    f.close()

def viz_textbb(name,text_im, charBB_list, wordBB,senBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1,figsize=(15,15))
    plt.imshow(text_im)
    H,W = text_im.shape[:2]

    # plot the character-BB:
    # for i in range(len(charBB_list)):
    #     bbs = charBB_list[i]
    #     ni = bbs.shape[-1]
    #     for j in range(ni):
    #         bb = bbs[:,:,j]
    #         bb = np.c_[bb,bb[:,0]]
    #         plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # # plot the word-BB:
    # for i in range(wordBB.shape[-1]):
    #     bb = wordBB[:,:,i]
    #     bb = np.c_[bb,bb[:,0]]
    #     plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
    #     # visualize the indiv vertices:
    #     vcol = ['r','g','b','k']
    #     for j in range(4):
    #         plt.scatter(bb[0,j],bb[1,j],color=vcol[j])

    for i in range(senBB.shape[-1]):
        bb = senBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in range(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    # plt.show(block=False)
    plt.savefig('img/' + name + '.jpg')

def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    print ("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
    for k in dsets:
        rgb = db['data'][k][...]
        # charBB = db['data'][k].attrs['charBB']
        # wordBB = db['data'][k].attrs['wordBB']
        senBB = db['data'][k].attrs['senBB']
        txt = db['data'][k].attrs['txt']

        # viz_textbb(k,rgb, [charBB], wordBB,senBB)
        # viz_textbb(k,rgb,None,None ,senBB)
        cre_output(k,rgb, senBB, txt)

        print ("image name        : ", colorize(Color.RED, k, bold=True))
        # print ("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
        # print ("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
        print ("  ** text         : ", colorize(Color.GREEN, txt))

        # if 'q' in input("next? ('q' to exit) : "):
        #     break
    db.close()

if __name__=='__main__':
    main('results/SynthText.h5')

