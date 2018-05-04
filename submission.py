import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import skimage
import sys
import tensorflow as tf

def prediction():
    smooth=1.
    TEST_PATH = './input/stage2_test_final/'
    # TEST_PATH = './input/stage1_test/'
    # train_ids = next(os.walk(TRAIN_PATH))[1]
    test_ids = next(os.walk(TEST_PATH))[1]

    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # Loss funtion
    def dice_coef_loss(y_true, y_pred):
        return -dice_coef(y_true, y_pred)

    def rle_encoding(x):
        dots = np.where(x.T.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b>prev+1): run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        return run_lengths

    def prob_to_rles(x, cutoff=0.5):
        lab_img = label(x > cutoff)
        for i in range(1, lab_img.max() + 1):
            yield rle_encoding(lab_img == i)

    # Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
    def mask_to_rle(preds_test_upsampled):
        new_test_ids = []
        rles = []
        for n, id_ in enumerate(test_ids):
            rle = list(prob_to_rles(preds_test_upsampled[n]))
            rles.extend(rle)
            new_test_ids.extend([id_] * len(rle))
        return new_test_ids,rles
    def reflect(img):
        shift=10
        reflect = cv2.copyMakeBorder(img,shift,shift,shift,shift,cv2.BORDER_REFLECT)
        return reflect

    def read_test_data(IMG_WIDTH=256-20,IMG_HEIGHT=256-20,IMG_CHANNELS=3):
        sizes_test = []
        print('\nGetting and resizing test images ... ')
        sys.stdout.flush()
        if os.path.isfile("numpy_data/test_img.npy") and os.path.isfile("numpy_data/test_size.npy"):
            print("Test file loaded from memory")
            X_test = np.load("numpy_data/test_img.npy")
            sizes_test = np.load("numpy_data/test_size.npy")
            return X_test,sizes_test
        X_test = np.zeros((len(test_ids), IMG_HEIGHT+20, IMG_WIDTH+20, IMG_CHANNELS), dtype=np.uint8)  
        for n, id_ in tqdm(enumerate(test_ids)):
            path = TEST_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')
            print (img.shape,"img shape")
            if len(img.shape)!=3:
                img=cv2.merge((img,img,img))
            img=img[:,:,:IMG_CHANNELS]    
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            img=img.astype(np.uint8)
            img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
            img=reflect(img)
            X_test[n] = img
        np.save("numpy_data/test_img",X_test)
        np.save("numpy_data/test_size",sizes_test)
        print (X_test.shape, "test data size")

        return X_test,sizes_test
        

    test_img,test_img_sizes = read_test_data()
    print (test_img.shape, "image shape")
    u_net = load_model('model-dsbowl2018-1_eroded.h5', custom_objects={'dice_coef': dice_coef})
    test_mask = u_net.predict(test_img,verbose=1)
    test=[]
    for i,mask in enumerate(test_mask):
        mask=mask[10:mask.shape[0]-10,10:mask.shape[1]-10]
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations = 1)
        test.append(mask)
    test_mask=test
    test_mask=np.array(test_mask)
    cv2.imwrite("original.jpg",test_img[1])
    cv2.imwrite("mask.jpg",test_mask[1]*255)

    # Create list of upsampled test masks
    test_mask_upsampled = []
    for i in range(len(test_mask)):
        test_mask_upsampled.append(resize(np.squeeze(test_mask[i]),
                                           (test_img_sizes[i][0],test_img_sizes[i][1]), 
                                           mode='constant', preserve_range=True))


    test_ids,rles = mask_to_rle(test_mask_upsampled)

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv('sub-dsbowl2018_eroded_dilated.csv', index=False)
    print("Data saved")

if __name__ == '__main__':
    prediction()