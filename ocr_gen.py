import os
from synthgen import *
from common import *
from functools import reduce
import re
from time import time
from data_provider import DateProvider
import glob
import torch
import csv
'''
Edit form train_ocr
'''
import collections
# import editdistance as ed
import os, sys
import numpy as np
import cv2
import math
import torch.nn.functional as F
import glob
import csv
import random
import editdistance
import torch
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from torchvision.utils import save_image
device = 'cpu'

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
  v = torch.from_numpy(x).type(dtype)
  if is_cuda:
      v = v.cuda()
  return v

def is_noise(image):
    # image (h,w,c) c=3
    from skimage.restoration import estimate_sigma
    noise_sigma = estimate_sigma(image, multichannel=True, average_sigmas=True)
    a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_1 = cv2.Laplacian(a, cv2.CV_64F).var()
    print(noise_sigma)
    print(noise_1)
    if noise_sigma > 1.2 and noise_1 >1000:
        return False
    else:
        return True



def load_gt(p, is_icdar=False):
    '''
    load annotation from the text file,
    :param p:
    :return:
    '''
    text_polys = []
    text_gts = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32), text_gts
    with open(p, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for line in reader:
            # strip BOM. \ufeff for python3,	\xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            # cls = 0
            gt_txt = ''
            delim = ''
            start_idx = 8
            if is_icdar:
                start_idx = 8

            for idx in range(start_idx, len(line)):
                gt_txt += delim + line[idx]
                delim = ','

            text_polys.append([x4, y4, x1, y1, x2, y2, x3, y3])
            text_line = gt_txt.strip()

            text_gts.append(text_line)

        return np.array(text_polys, dtype=np.float), text_gts


def ocr_image( im_data, boxr,w,h, target_h):
    # boxo = detection
    # boxr = boxo[0:8].reshape(-1, 2)

    center = (boxr[0, :] + boxr[1, :] + boxr[2, :] + boxr[3, :]) / 4

    # dw = boxr[2, :] - boxr[1, :]
    # dw2 = boxr[0, :] - boxr[3, :]
    # dh = boxr[1, :] - boxr[0, :]
    # dh2 = boxr[3, :] - boxr[2, :]
    #
    # h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + 1
    # h2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1]) + 1
    # h = (h + h2) / 2
    # print(h)
    # w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
    # w2 = math.sqrt(dw2[0] * dw2[0] + dw2[1] * dw2[1])
    # w = (w + w2) / 2

    input_W = im_data.size(3)
    input_H = im_data.size(2)

    # scale = target_h / max(1, h)

    # target_gw = int(w * scale + target_h / 4)
    target_gw = int(w + h / 4)

    target_gw = max(8, int(round(target_gw / 4)) * 4)
    xc = center[0]
    yc = center[1]
    w2 = w
    h2 = h

    angle = math.atan2((boxr[2][1] - boxr[1][1]), boxr[2][0] - boxr[1][0])
    angle2 = math.atan2((boxr[3][1] - boxr[0][1]), boxr[3][0] - boxr[0][0])
    angle = (angle + angle2) / 2

    # show pooled image in image layer
    scalex = (w2 + h2 / 4) / input_W
    scaley = h2 / input_H

    # th11 = scalex * math.cos(angle)
    # th12 = -math.sin(angle) * scaley
    # th13 = (2 * xc - input_W - 1) / (
    #             input_W - 1)  # * torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)
    #
    # th21 = math.sin(angle) * scalex
    # th22 = scaley * math.cos(angle)
    # th23 = (2 * yc - input_H - 1) / (
    #             input_H - 1)  # * torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)
    th11 = scalex * math.cos(angle)
    th12 = -math.sin(angle) * scaley * input_H / input_W
    th13 = (2 * xc - input_W - 1) / (input_W - 1)

    th21 = math.sin(angle) * scalex * input_W / input_H
    th22 = scaley * math.cos(angle)
    th23 = (2 * yc - input_H - 1) / (input_H - 1)

    t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
    t = torch.from_numpy(t).type(torch.FloatTensor)
    t = t.to(device)
    theta = t.view(-1, 2, 3)

    grid = F.affine_grid(theta, torch.Size((1, 3, int(h), int(target_gw))))
    # grid = F.affine_grid(theta, torch.Size((1, 3, int(h), int(target_gw))))

    x = F.grid_sample(im_data, grid)

    # mask_gray = cv2.normalize(src=x.data.numpy().squeeze(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
    #                           dtype=cv2.CV_8UC1)
    # cv2.imshow('abc', mask_gray.transpose(1, 2, 0))
    # cv2.waitKey(0)
    # cv2.write('đasad',x)
    #
    return x


def ocr_generate(img_name, img, fe):
    # img_name = imagess[i]
    base_nam = os.path.basename(img_name)
    res_gt = base_nam.replace(".jpg", '.txt').replace(".png", '.txt')
    res_gt = '{0}/gt_{1}'.format(images_dir, res_gt)
    if not os.path.exists(res_gt):
        res_gt = base_nam.replace(".jpg", '.txt').replace("_", "")
        res_gt = '{0}/gt_{1}'.format(images_dir, res_gt)
        if not os.path.exists(res_gt):
            print('missing! {0}'.format(res_gt))
            gt_rect, gt_txts = [], []
        # continue
    gt_rect, gt_txts = load_gt(res_gt)
    # img = cv2.imread(img_name)
    img = np.expand_dims(img, axis=0)
    im_data = np_to_variable(img, is_cuda=False).permute(0, 3, 1, 2)
    for idx, box in enumerate(gt_rect):
        boxr = box[0:8].reshape(-1, 2)

        dw = boxr[2, :] - boxr[1, :]
        dw2 = boxr[0, :] - boxr[3, :]
        dh = boxr[1, :] - boxr[0, :]
        dh2 = boxr[3, :] - boxr[2, :]

        h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + 1
        h2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1]) + 1
        h = (h + h2) / 2

        w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
        w2 = math.sqrt(dw2[0] * dw2[0] + dw2[1] * dw2[1])
        w = (w + w2) / 2
        w_cha = w / len(gt_txts[idx])

        # if h > 32 and w > 85 and w_cha > 10:
        #     # print('h',h)
        #     # print('chia',w/len(gt_txts[idx]))
        #     # print(gt_txts[idx])
        #     imgocr = ocr_image(im_data, boxr, w, h, 48)
        #     # str = gt_txts[idx]
        #     name_im = base_nam.replace(".jpg", '_') + str(idx) + '.jpg'
        #     string = name_im + ', ' + '"' + gt_txts[idx] + '"'
        #     fe.write(string + '\n')
        #     # save_image(imgocr,'ocr/'+ name_im)
        #     cv2.imwrite('ocr/'+ name_im,imgocr.permute(0, 2, 3, 1).data.numpy().squeeze())
        #     # shutil.move(re.sub(r'\\', r'/', line), dest_folder + b)


        if h > 36:
            # print('h',h)
            # print('chia',w/len(gt_txts[idx]))
            # print(gt_txts[idx])
            imgocr = ocr_image(im_data, boxr, w, h, 48)
            # img_np = imgocr.permute(0, 2, 3, 1).data.numpy().squeeze()
            mask_gray = cv2.normalize(src=imgocr.permute(0, 2, 3, 1).data.numpy().squeeze(), dst=None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8UC1)

            # if is_noise(mask_gray) == False:
                # str = gt_txts[idx]
            name_im = base_nam.replace(".jpg", '_') + str(idx) + 'nb.jpg'
            string = name_im + ', ' + '"' + gt_txts[idx] + '"'
            fe.write(string + '\n')
            # save_image(imgocr,'ocr/'+ name_im)
            cv2.imwrite('ocr/'+ name_im,mask_gray)
            # cv2.imwrite('ocr/'+ name_im,imgocr.permute(0, 2, 3, 1).data.numpy().squeeze())
            # shutil.move(re.sub(r'\\', r'/', line), dest_folder + b)



images_dir = 'masks'
# images = glob.glob(os.path.join(images_dir, '*.jpg'))
# imagess = np.asarray(images)
# index = np.arange(0, imagess.shape[0])
# fe = open('ocr_gt.txt','w',encoding='utf-8')


# for i in index:
#     img_name = imagess[i]
#     base_nam = os.path.basename(img_name)
#     res_gt = base_nam.replace(".jpg", '.txt').replace(".png", '.txt')
#     res_gt = '{0}/gt_{1}'.format(images_dir, res_gt)
#     if not os.path.exists(res_gt):
#         res_gt = base_nam.replace(".jpg", '.txt').replace("_", "")
#         res_gt = '{0}/gt_{1}'.format(images_dir, res_gt)
#         if not os.path.exists(res_gt):
#             print('missing! {0}'.format(res_gt))
#             gt_rect, gt_txts = [], []
#         # continue
#     gt_rect, gt_txts = load_gt(res_gt)
#     img = cv2.imread(img_name)
#     img = np.expand_dims(img, axis=0)
#     im_data = np_to_variable(img, is_cuda=False).permute(0, 3, 1, 2)
#     for idx, box in enumerate(gt_rect):
#         boxr = box[0:8].reshape(-1, 2)
#
#         dw = boxr[2, :] - boxr[1, :]
#         dw2 = boxr[0, :] - boxr[3, :]
#         dh = boxr[1, :] - boxr[0, :]
#         dh2 = boxr[3, :] - boxr[2, :]
#
#         h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + 1
#         h2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1]) + 1
#         h = (h + h2) / 2
#
#         w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
#         w2 = math.sqrt(dw2[0] * dw2[0] + dw2[1] * dw2[1])
#         w = (w + w2) / 2
#         w_cha = w / len(gt_txts[idx])
#
#         if h > 30 and w > 80 and w_cha > 10 :
#             # print('h',h)
#             # print('chia',w/len(gt_txts[idx]))
#             # print(gt_txts[idx])
#             imgocr = ocr_image(im_data, boxr, w, h, 48)
#             # str = gt_txts[idx]
#
#             string = base_nam.replace(".jpg", '_') + str(idx)+ '.jpg' + ', ' + '"' + gt_txts[idx] + '"'
#             fe.write(string + '\n')
#             # shutil.move(re.sub(r'\\', r'/', line), dest_folder + b)
#
# for i in index:
#     ocr_generate(imagess[i],fe)




