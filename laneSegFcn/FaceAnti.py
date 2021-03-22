# -*- coding: utf-8 -*-
# @Time    : 2021/2/6 14:04
# @Author  : Zeqi@@
# @FileName: FaceAnti.py
# @Software: PyCharm

import os
import torch
import cv2
import numpy as np
from model.FaceBagNet_model_A import Net
from collections import OrderedDict
import torch.nn.functional as F
from imgaug import augmenters as iaa

pwd = os.path.abspath('./')
RESIZE_SIZE=112

def TTA_36_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_up = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))

    return images
#
# def TTA_36_cropps(image, target_shape=(32, 32, 3)):
#     image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))
#
#     width, heigth, d = image.shape
#     target_w, target_h, d = target_shape
#
#     start_x = (width - target_w)//2
#     start_y = (heigth - target_h)//2
#
#     starts = [[start_x, start_y],
#               [start_x - target_w, start_y],
#               [start_x, start_y-target_w],
#               [start_x + target_w, start_y],
#               [start_x, start_y + target_w],
#
#               [start_x + target_w, start_y + target_w],
#               [start_x - target_w, start_y - target_w],
#               [start_x - target_w, start_y + target_w],
#               [start_x + target_w, start_y - target_w],
#               ]
#
#     images = []
#
#     for start_index in starts:
#         image_ = image.copy()
#         x, y = start_index
#
#         if x < 0:
#             x = 0
#         if y < 0:
#             y= 0
#
#         if x + target_w >= RESIZE_SIZE:

class FaceAnti:
    def __init__(self):
        # 准备模型
        self.net = Net(num_class=2, is_first_bn=True) # 网络结构加载
        model_path = 'model_A_color_48/checkpoint/global_min_acer_model.pth' # 权重加载
        if torch.cuda.is_available():
            state_dict = torch.load(model_path, map_location='cuda')
        else:
            state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            name = k[7:] # remove 'module'
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        self.net.eval() # bn 层输出锁定

    def classify(self, color):
        return self.detect(color)

    def detect(self, color):
        color = cv2.resize(color, (RESIZE_SIZE, RESIZE_SIZE))

        def color_augmentor(image, target_shape=(64, 64, 3), is_infer=False):
            if is_infer:
                augment_img = iaa.Sequential([iaa.Fliplr(0)])

            image = augment_img.augment_image(image)
            image = TTA_36_cropps(image, target_shape)
            return image

        color = color_augmentor(color, target_shape=(64, 64, 3), is_infer=True)

        n = len(color)
        color = np.concatenate(color, axis=0)

        image = color
        image = np.transpose(image, (0, 3, 1, 2))
        image = image.astype(np.float32)
        image = image/255.0
        input_image = torch.FloatTensor(image)
        if (len(input_image.size()) == 4) and torch.cuda.is_available():
            input_image = input_image.unsqueeze(0).cuda()
        elif (len(input_image.size())==4) and not torch.cuda.is_available():
            input_image = input_image.unsqueeze(0)

        b,  n, c, w, h = input_image.size()
        input_image = input_image.view(b*n, c, w, h)
        if torch.cuda.is_available():
            input_image = input_image.cuda()

        with torch.no_grad():
            logit, _, _ = self.net(input_image)
            logit = logit.view(b, n, 2)
            logit = torch.mean(logit, dim=1, keepdim=False)
            prob = F.softmax(logit, 1)

        print('probabilistic: ', prob)
        print('predict: ', np.argmax(prob.detach().cpu().numpy()))
        return np.argmax(prob.detach().cpu().numpy())

if __name__ =='__main__':
    FA = FaceAnti()
    img = cv2.imread('1.jpg', 1)
    FA.detec(img)