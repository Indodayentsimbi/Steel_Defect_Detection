#%%
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'modules'))
# data structures:
import pandas as pd
import numpy as np
from collections import Counter
import math
# visuals:
import seaborn as sns
import matplotlib.pyplot as plt
# images:
import cv2
# helper functions:
from mask_functions import mask2rle,rle2mask

from sklearn.model_selection import train_test_split

from loss_functions import dice_coef,dice_loss,combined_loss

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,concatenate,Conv2DTranspose,BatchNormalization,Dropout
from keras.losses import binary_crossentropy

%matplotlib inline

#%%
train_data = pd.read_csv(os.path.join('./data','train.csv'))
train_data.sort_values(by='ImageId_ClassId',inplace=True)
train_data.head()

#%%
train_data.info()

#%%
train_data.shape

#%%
image_label_dict = dict()
for row in train_data.iterrows():
    image,label = row[1][0].split('_')
    if image_label_dict.get(image,0) == 0:
        image_label_dict[image] = [int(label)]
        if not isinstance(row[1][1],str):
            image_label_dict[image].pop()
    else:
        image_label_dict.get(image).append(int(label))
        if not isinstance(row[1][1],str):
            image_label_dict[image].pop()

for key,value in image_label_dict.items():
    image_label_dict[key] = [value,len(value)]

#%%
train_data_visuals = pd.DataFrame({})
train_data_visuals['image_id'] = list(image_label_dict.keys())
train_data_visuals['defect_types'] = train_data_visuals['image_id'].apply(lambda x: image_label_dict.get(x)[0])
train_data_visuals['defect_count'] = train_data_visuals['image_id'].apply(lambda x: image_label_dict.get(x)[1])
train_data_visuals['image_width'] = train_data_visuals['image_id'].apply(lambda x: cv2.imread(os.path.join('./data/train_imgs',x)).shape[1])
train_data_visuals['image_height'] = train_data_visuals['image_id'].apply(lambda x: cv2.imread(os.path.join('./data/train_imgs',x)).shape[0])

#%%
EncodedPixels = dict()
for image in image_label_dict.keys():
    counter = 0
    for e in list(train_data[train_data['ImageId_ClassId'].apply(lambda x: x.split('_')[0])==image]['EncodedPixels'].values):
        if isinstance(e,str):
            if EncodedPixels.get(image,0) == 0:
                EncodedPixels[image] = [e]
            else:
                EncodedPixels.get(image).append(e)
        else:
            counter += 1
            if counter == 4:
                EncodedPixels[image] = ''

#%%
train_data_visuals = train_data_visuals.merge(pd.DataFrame(list(EncodedPixels.items()),columns=['image_id','encodedpixels']),how='inner',on='image_id')

#%%
train_data_visuals.head()

#%%
train_data_visuals.shape

#%%
print('Ratio of material that do not have any defects: {} %'.format(round((dict(Counter(train_data_visuals['defect_count'].apply(lambda x: 1 if x > 0 else 0))).get(0)/train_data_visuals.shape[0])*100,2)))
print('Ratio of material that do have defects: {} %'.format(round((dict(Counter(train_data_visuals['defect_count'].apply(lambda x: 1 if x > 0 else 0))).get(1)/train_data_visuals.shape[0])*100,2)))

#%%
sns.countplot(train_data_visuals['defect_count'])
plt.title('Histogram on the number of defects')
plt.xlabel('Number of defects')
plt.show()

#%%
Counter(train_data_visuals['defect_count'])

#%%
X = list()
Y = list()
for x,y in sorted([(value,key) for key,value in dict(Counter(train_data_visuals['defect_types'].astype(str))).items()],reverse=False):
    X.append(x)
    Y.append(y)

plt.barh(y=Y,width=X)
plt.title('Distribution of the defect type permutation')
plt.xlabel('count')
plt.ylabel('defect combinations')
plt.show()
sorted([(value,key) for key,value in dict(Counter(train_data_visuals['defect_types'].astype(str))).items()],reverse=False)

#%%
count_per_defect = list()
for item in list(train_data_visuals['defect_types']):
    count_per_defect += item

sns.countplot(count_per_defect)
plt.title('Histogram on the count per defect type')
plt.xlabel('Defect type')
plt.show()

#%%
Counter(count_per_defect)

#%%
def mask2rle(mask):
    '''
    mask: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    if len(mask_rle) == 0:
        return np.zeros(shape[0]*shape[1], dtype=np.uint8).reshape(shape).T
    else:    
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 1
        mask = mask.reshape(shape).T
        return mask

try:
    assert '1 3 10 5' == mask2rle(rle2mask('1 3 10 5'))
    assert '1 1' == mask2rle(rle2mask('1 1'))
    print('Function mask is good')
except AssertionError as e:
    print('Error in function mask')        

#%%
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * np.sum(intersection) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return 1. - score

def combined_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

#%%
color_scheme = {1:(255,0,0),2:(0,255,0),3:(0,0,255),4:(255,0,255)}
for i,c in color_scheme.items():
    plt.subplot(1,4,i)
    plt.axis('off')
    plt.imshow(np.ones((20, 20, 3), dtype=np.uint8)*c)
    plt.title('Defect type {}'.format(i))
plt.show()

#%%
def mask_to_contours(image, mask, color):
    """ Converts a mask to contours using OpenCV and draws it on the image
    """ 
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2)        
    return image

def apply_masks(file_name, encodings, labels):
    if not isinstance(encodings,list):
        encodings = list()
    if not isinstance(labels,list):
        labels = list()
    # reading in the image
    image = cv2.imread(os.path.join('./data/train_imgs',file_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = [rle2mask(rle) for rle in encodings]

    for mask,label in zip(masks,labels):
        image = mask_to_contours(image=image, mask=mask, color=color_scheme.get(label,(0,0,0)))
    return image

#%%
for permutation in np.unique(train_data_visuals['defect_types'].values):      
    data = train_data_visuals[train_data_visuals['defect_types'].apply(lambda x: str(x)) == str(permutation)]
    
    if data.shape[0] > 2:
        data = data.sample(n=3)
    elif data.shape[0] == 2:
        data = data.sample(n=2)
    else:
        data = data.sample(n=1)               
    
    for row in data.iterrows():
        plt.figure(figsize=(10,10))
        plt.axis('off')
        plt.title('Image: ' + str(row[1][0]))
        plt.imshow(apply_masks(file_name=row[1][0],encodings=row[1][5],labels=row[1][1]))
        plt.xlabel('defect types: ' + str(row[1][1]))
        plt.show()
#%%
del train_data

#%%
training_set,validation_set = train_test_split(train_data_visuals,
                                                test_size=0.2,
                                                shuffle=True)

print('Train data size: {}'.format(training_set.shape))
print('Validation data size: {}'.format(validation_set.shape))

#%%
dict(Counter(training_set['defect_types'].apply(lambda x: str(x))))

#%%
dict(Counter(validation_set['defect_types'].apply(lambda x: str(x))))

#%%
def data_generator(batch_size,mode,data,size):
    data_set_index = data.index.values
    start = 0
    batch = batch_size
    global batch_X
    global batch_Y
    batch_X = list()
    batch_Y = list() 
    while True: 
        if mode == 'training':
            while batch < data_set_index.shape[0]:
                batch_images = np.array(data.iloc[start:batch,[0,5]]['image_id'].apply(lambda x:cv2.imread(filename=os.path.join('./data/train_imgs',x))))
                batch_images = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),batch_images))) #get the channel ordering right
                batch_images = np.array(list(map(lambda x: cv2.resize(x,(size,size)),batch_images)))
                
                batch_masks = np.array(data.iloc[start:batch,[0,5]]['encodedpixels'].apply(lambda x: np.array([rle2mask(rle) for rle in list(x)]) if len(x) > 0 else np.array([np.zeros(shape=(256,1600))])))
                batch_masks = [x.transpose((1,2,0)) for x in batch_masks]
                batch_masks = list(map(lambda x: cv2.resize(x,(size,size)),batch_masks))

                batch_X.append(batch_images)
                batch_Y.append(batch_masks)
                print('-------------')
                yield np.array(batch_X),batch_Y
                start = batch
                batch += batch_size 
                batch_images, batch_masks = np.array([])

        if mode == 'prediction':
            while batch < data_set_index.shape[0]:
                batch_images = np.array(data.iloc[start:batch,[0,5]]['image_id'].apply(lambda x:cv2.imread(filename=os.path.join('./data/test_imgs',x))))
                batch_images = np.array(list(map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),batch_images))) #get the channel ordering right
                batch_images = np.array(list(map(lambda x: cv2.resize(x,(size,size)),batch_images)))

                batch_X.append(batch_images)
                print('-------------')
                yield np.array(batch_X)
                start = batch
                batch += batch_size 
                batch_images = np.array([]) 


#%%
batch_size = 30
training_batch_generator = data_generator(batch_size=batch_size,mode='training',data=training_set,size=256)
validation_batch_generator = data_generator(batch_size=batch_size,mode='training',data=validation_set,size=256)

#%%
next(training_batch_generator)

#%%
print(batch_X[0][0].shape)
print(batch_Y[0][0].shape)
#%%
print(batch_X[0].shape)
print(batch_Y[0].shape)

#%%
inputs = Input(shape=(256,256,3))

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=inputs, outputs=outputs)

#%%
model.summary()

#%%
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

#%%
H = model.fit_generator(generator=training_batch_generator, 
                        steps_per_epoch=math.ceil(training_set.shape[0]/batch_size), 
                        epochs=3, 
                        verbose=1, 
                        callbacks=None, 
                        validation_data=validation_batch_generator, 
                        validation_steps=math.ceil(validation_set.shape[0]/batch_size)
                        )

#%%
