import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import skimage.transform as trans
from keras.layers import *
from keras.optimizers import *

def create_pascal_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) > len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]


def vis_segmentation(image, seg_map,target_size,path,index):

    seg_image = label_to_color_image(seg_map)
    plt.figure()
    plt.imshow(seg_image)
    image = trans.resize(image,target_size)
    plt.imshow(image,alpha=0.5)
    plt.axis('off')
    # plt.show()
    plt.savefig('data/results/'+path+'_seg_vis/'+str(index))

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def cal_accuracy(label_image,test_image):
    assert test_image.shape == label_image.shape
    test_image = test_image.astype(np.float64)
    label_image= label_image.astype(np.float64)
    true_1 = label_image==1
    true_0 = label_image==0
    all_1 =  np.sum(label_image[true_1])
    total_pixels = label_image.shape[0]*label_image.shape[1]
    TP = np.sum(test_image[true_1])
    FP = np.sum(test_image[true_0])
    FN = all_1-TP
    TN = (total_pixels-all_1)-FP
    acc =(TN+TP)/(FN+FP+TN+TP)
    dsc = (2*TP)/(2*TP+FP+FN)
    # print(label_image)
    return (TP/total_pixels, TN/total_pixels, FP/total_pixels, FN/total_pixels, acc, dsc)
