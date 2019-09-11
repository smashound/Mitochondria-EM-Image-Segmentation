from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
import skimage.io as io
import skimage.transform as trans
GAMMA = 0.5
def preprocess_gamma_hist(imgs, gamma=GAMMA):
    invGamma = 1.0/gamma
    #build the gamma lookup table for color correctness (grayscale)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    #apply gamma correction that controls the overall brightness
    new_imgs = np.empty(imgs.shape)
    new_imgs = cv2.LUT(np.array(imgs, dtype = np.uint8), table)
    #apply the histogram equalization to improve the contrast
    new_img = cv2.equalizeHist(new_imgs)
    return new_img
class Data_generator():
    def __init__(self):
        self.train_path = 'data/train'
        self.val_path = 'data/val'
        self.test_path = 'data/test'
    def train_gen(self,data_gen_args,batch_size=2,target_size=(384,512)):
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        image_generator = image_datagen.flow_from_directory(
        self.train_path,
        classes = ['raw'],
        class_mode = None,
        color_mode = 'grayscale',
        target_size = target_size,
        batch_size = batch_size,
        seed = 1)
        mask_generator = mask_datagen.flow_from_directory(
        self.train_path,
        classes = ['label'],
        class_mode = None,
        color_mode = 'grayscale',
        target_size = target_size,
        batch_size = batch_size,
        seed = 1)
        train_generator = zip(image_generator, mask_generator)
        for (img,mask) in train_generator:
            # img[0,:,:,0] = preprocess_gamma_hist(img[0,:,:,0]).astype(np.float)
            img,mask = img/255,mask/255
            yield (img,mask)
    def val_gen(self,target_size=(384,512)):
        path = self.val_path
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
        image_generator = image_datagen.flow_from_directory(
        path,
        classes = ['raw'],
        class_mode = None,
        color_mode = 'grayscale',
        target_size = target_size,
        seed = 1)
        mask_generator = mask_datagen.flow_from_directory(
        path,
        classes = ['label'],
        class_mode = None,
        color_mode = 'grayscale',
        target_size = target_size,
        seed = 1)
        train_generator = zip(image_generator, mask_generator)
        for (img,mask) in train_generator:
            # img[0,:,:,0] = preprocess_gamma_hist(img[0,:,:,0]).astype(np.float)
            img,mask = img/255,mask/255
            yield (img,mask)
    def test_gen(self,target_size=(384,512)):
        path = self.test_path
        raw_path  = os.path.join(path, 'raw')
        mask_path = os.path.join(path, 'label')
        raw_images = os.listdir(raw_path)
        mask_images = os.listdir(mask_path)
        total = len(raw_images)
        imgs = np.ndarray((total,)+ target_size+(1,))
        masks = np.ndarray((total,)+ target_size)
        for i in range(total):
            img = io.imread(os.path.join(raw_path,raw_images[i]),as_gray = True)
            mask = io.imread(os.path.join(mask_path,mask_images[i]),as_gray = True)
            img = trans.resize(img,target_size)
            mask = trans.resize(mask,target_size)
            mask = np.reshape(mask,(1,)+mask.shape)
            img = np.reshape(img,img.shape+(1,))
            img = np.reshape(img,(1,)+img.shape)
            # img[0,:,:,0] = preprocess_gamma_hist(img[0,:,:,0]).astype(np.float)
            imgs[i]=img
            masks[i]=mask
        return imgs,masks


