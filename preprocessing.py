import os
import random
import sys
import warnings
import numpy as np
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
import matplotlib.pyplot as plt
from unet_model import train_model
from submission import prediction 

seed = 42
random.seed = seed
np.random.seed = seed

# Data Path
TRAIN_PATH = 'input/stage1_train/'
TEST_PATH = 'input/stage2_test_final/'

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

def reflect(img):
    shift=10
    reflect = cv2.copyMakeBorder(img,shift,shift,shift,shift,cv2.BORDER_REFLECT)
    return reflect

def filters(img):
    kernel = np.ones((5,5),np.float32)/25
    blur1 = cv2.filter2D(img,-1,kernel)
    blur2 = cv2.blur(img,(11,11))
    blur3 = cv2.GaussianBlur(img,(31,31),0)
    blur4 = cv2.medianBlur(img,11)
    blur5 = cv2.bilateralFilter(img,9,75,75)
    gamma = adjust_gamma(img, gamma=1.5)
    return blur1,blur2,blur3,blur4,blur5,gamma 

def adjust_gamma(image,gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def rotate(img):
    shape=img.shape[2]
    if shape==1:
        img=np.squeeze(img)        
    horizontal_img = cv2.flip( img, 0 )
    vertical_img = cv2.flip( img, 1 )
    both_img = cv2.flip( img, -1 )
    if shape==1:
        horizontal_img=np.expand_dims(horizontal_img,-1)
        vertical_img=np.expand_dims(vertical_img,-1)
        both_img=np.expand_dims(both_img,-1)
    return horizontal_img,vertical_img,both_img

def elastic_transform(image,mask, alpha, sigma, random_state):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    shape = image.shape[:2]
    r,g,b=cv2.split(image)
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    r=map_coordinates(r, indices, order=1).reshape(shape)
    g=map_coordinates(g, indices, order=1).reshape(shape)
    b=map_coordinates(b, indices, order=1).reshape(shape)
    image=cv2.merge((r,g,b))
    mask=np.squeeze(mask)
    mask=map_coordinates(mask, indices, order=1).reshape(shape)
    mask= np.expand_dims(mask, axis=-1)
    return image,mask

def split_data(X_train,Y_train):
    red_train=[]
    red_mask=[]
    black_train=[]
    black_mask=[]
    for im,mask in zip(X_train,Y_train):
        if np.sum(im[:,:,2])/(im.shape[1]*im.shape[0])>80:
            red_train.append(im)
            red_mask.append(mask)
            rs= np.random.RandomState(None)
            image_elastic,mask_elastic=elastic_transform(im,mask,1800,50,rs)
            black_train.append(image_elastic)
            black_mask.append(mask_elastic)
            blur1,blur2,blur3,blur4,blur5,gamma =filters(im)
            red_train.append(blur1)
            red_mask.append(mask)
            red_train.append(blur2)
            red_mask.append(mask)
            red_train.append(blur3)
            red_mask.append(mask)
            red_train.append(blur4)
            red_mask.append(mask)
            red_train.append(blur5)
            red_mask.append(mask)
            red_train.append(gamma)
            red_mask.append(mask)
            filters_=[blur1,blur2,blur3,blur4,blur5,gamma]
            for image in filters_:
                horizontal_img,vertical_img,both_img=rotate(image)
                red_train.append(horizontal_img)
                red_train.append(vertical_img)
                red_train.append(both_img)
                horizontal_mask,vertical_mask,both_mask=rotate(mask)
                red_mask.append(horizontal_mask)
                red_mask.append(vertical_mask)
                red_mask.append(both_mask)
        else:
            black_train.append(im)
            black_mask.append(mask)    
            rs= np.random.RandomState(None)
            image_elastic,mask_elastic=elastic_transform(im,mask,1800,50,rs)
            black_train.append(image_elastic)
            black_mask.append(mask_elastic)
            blur1,blur2,blur3,blur4,blur5,gamma =filters(im)
            black_train.append(blur1)
            black_mask.append(mask)
            black_train.append(blur2)
            black_mask.append(mask)
            black_train.append(blur3)
            black_mask.append(mask)
            black_train.append(blur4)
            black_mask.append(mask)
            black_train.append(blur5)
            black_mask.append(mask)
            black_train.append(gamma)
            black_mask.append(mask)
            filters_=[blur1,blur2,blur3,blur4,blur5,gamma]
            for image in filters_:
                horizontal_img,vertical_img,both_img=rotate(image)
                black_train.append(horizontal_img)
                black_train.append(vertical_img)
                black_train.append(both_img)
                horizontal_mask,vertical_mask,both_mask=rotate(mask)
                black_mask.append(horizontal_mask)
                black_mask.append(vertical_mask)
                black_mask.append(both_mask)
    for i,(im,mask) in enumerate(zip(red_train,red_mask)):
        red_train[i]=reflect(im)
     #    kernel = np.ones((3,3),np.uint8)
    	# mask = cv2.erode(mask,kernel,iterations = 1)
        mask=reflect(mask)
        red_mask[i]=mask.astype(dtype=np.bool)
        
    for i,(im,mask) in enumerate(zip(black_train,black_mask)):
        black_train[i]=reflect(im)
     #    kernel = np.ones((3,3),np.uint8)
    	# mask = cv2.erode(mask,kernel,iterations = 1)
        mask=reflect(mask)
        black_mask[i]=mask.astype(dtype=np.bool)
        
    print (len(red_train), "total red train images", len(red_mask)," total mask")
    print (len(black_train), "total black train images", len(black_mask)," total mask")
    val_data_train=red_train[:int(len(red_train)*0.1)]+black_train[:int(len(black_train)*0.1)]  
    val_data_label=red_mask[:int(len(red_mask)*0.1)]+black_mask[:int(len(black_mask)*0.1)]  
    train_data=red_train[int(len(red_train)*0.1):]+black_train[int(len(black_train)*0.1):]
    train_label=red_mask[int(len(red_mask)*0.1):]+black_mask[int(len(black_mask)*0.1):]
    
    #shuffling training data
    c = list(zip(train_data, train_label))
    random.shuffle(c)
    train_data, train_label = zip(*c)
    train_data=np.array(train_data)
    train_label=np.array(train_label)    
    np.save("numpy_data/train_img",train_data)
    np.save("numpy_data/train_mask",train_label)
    np.save("numpy_data/val_data_train",val_data_train)
    np.save("numpy_data/val_data_label",val_data_label)
    
    print (train_data.shape, "X_ train size", train_label.shape, "Y_train size")
    return val_data_train,val_data_label,train_data,train_label

# Function read train images and mask return as nump array
def read_train_data(IMG_WIDTH=256-20,IMG_HEIGHT=256-20,IMG_CHANNELS=3):
    X_train=[]
    Y_train=[]
    print ('loading raw images and resizing ... ')
    sys.stdout.flush()
    if os.path.isfile("numpy_data/train_img_raw.npy") and os.path.isfile("numpy_data/train_mask_raw.npy"):
        print ("training images loaded from memory")
        X_train = np.load("numpy_data/train_img_raw.npy")
        Y_train = np.load("numpy_data/train_mask_raw.npy")
        return X_train,Y_train
    for n, id_ in tqdm(enumerate(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        
        img = cv2.resize(img,None, fx=float(IMG_WIDTH)/img.shape[1], fy=float(IMG_HEIGHT)/img.shape[0], interpolation = cv2.INTER_AREA)        
        X_train.append(img)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(cv2.resize(mask_, None, fx=float(IMG_WIDTH)/mask_.shape[1], fy=float(IMG_HEIGHT)/mask_.shape[0], interpolation = cv2.INTER_AREA)
    , axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train.append(mask)
    np.save("numpy_data/train_img_raw",X_train)
    np.save("numpy_data/train_mask_raw",Y_train)
    return X_train,Y_train

if __name__ == '__main__':
    x,y = read_train_data()
    val_data_train,val_data_label,train_data,train_label=split_data(x,y)
    # train_model()
    # prediction()
    
