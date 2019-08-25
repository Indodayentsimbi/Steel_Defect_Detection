#%%
import os
import cv2
import numpy as np
import math
from mask_functions import rle2mask,mask2rle
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#%%
def normalize_data(x):
    return x/255

#%%
image_generator = ImageDataGenerator(data_format='channels_last',
                                    rotation_range=90,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    zoom_range=0.2)
#%%
for image_name in os.listdir(path='./data/train_imgs'):#pass image from data sets:
    img = cv2.imread(filename=os.path.join('./data/train_imgs',image_name)) #read the image as an array of pixels
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #get the channel ordering right
    #mask = rle2mask(mask_rle=)
    img = np.array(img).transpose((1,0,2)) # fix the shape of the array -> (height-y,width-x,channel-z)

    prob_of_augmentation = np.random.rand()

    if prob_of_augmentation > 0.5:
        random_transform = image_generator.get_random_transform(img_shape=img.shape)
        img = image_generator.apply_transform(x=img,transform_parameters=random_transform)
        img = normalize_data(img)

    else:
        img = normalize_data(img)
#%%
img = cv2.imread(filename=os.path.join('./data/train_imgs','e006b532b.jpg')) #read the image as an array of pixels
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #get the channel ordering right
# img = img.transpose((1,0,2))
plt.imshow(img)
plt.show()
#%%
img.shape
# img = np.array([img])
# img.shape
#%%
returnv = image_generator.flow(x=img,batch_size=2)
#%%
k = 0
while k <=2:
    print('First yield')
    print(next(returnv).shape)
    k += 1
