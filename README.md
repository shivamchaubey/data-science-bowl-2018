# data-science-bowl-2018 LB 0.367  (https://www.kaggle.com/c/data-science-bowl-2018)
An algorithm to automate nucleus detection

This project is focused on detecting nuclei in a histology image. This project include three part preprocessing, training and post processing. 
U-Net architecture were used for segmentation.

Required python libraries: numpy,skimage,opencv3.3,scipy, tqdm,matplotlib,pandas, keras, tensorflow,   

Preprocessing steps: 
Data augmentation has been done on raw training data. 
1. Different blur techniques were used with different kerel sizes.
2. Elastic deformation.
3. Horizontal, vertical and horizontal-vertical flip.
4. Reflection of border upto ten pixel width.
5. Diferent types of images were used for Validation data. 
6. Erosion of training mask to separate touching nuclei.

Training data of size 256X256 pixels were prepared for training model.

Training model:
Trainging is performed in google colaboratory.
Unet-architecture were used with binary_crossentropy loss function. 
(https://arxiv.org/abs/1505.04597)

Post porcessing steps:
1. Predicted mask were cropped and resized to orginal training image.
2. Dilation were done to recover the actual size.
3. Hole filling on predicted mask
