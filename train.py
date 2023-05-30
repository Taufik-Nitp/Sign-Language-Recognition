# # Importing the Keras libraries and packages
# from keras.models import Sequential
# from keras.layers import Convolution2D
# from keras.layers import MaxPooling2D
# from keras.layers import Flatten
# from keras.layers import Dense , Dropout
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sz = 128
# # Step 1 - Building the CNN

# # Initializing the CNN
# classifier = Sequential()

# # First convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# # Second convolution layer and pooling
# classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
# #classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# # input_shape is going to be the pooled feature maps from the previous convolution layer
# #classifier.add(MaxPooling2D(pool_size=(2, 2)))

# # Flattening the layers
# classifier.add(Flatten())

# # Adding a fully connected layer
# classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dropout(0.40))
# classifier.add(Dense(units=96, activation='relu'))
# classifier.add(Dropout(0.40))
# classifier.add(Dense(units=64, activation='relu'))
# classifier.add(Dense(units=26, activation='softmax')) # softmax for more than 2

# # Compiling the CNN
# classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# # Step 2 - Preparing the train/test data and training the model
# classifier.summary()
# # Code copied from - https://keras.io/preprocessing/image/
# from keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory('/home/taufik/Desktop/SignLanguage/Sign-Language-to-Text/data/train',
#                                                  target_size=(sz, sz),
#                                                  batch_size=10,
#                                                  color_mode='grayscale',
#                                                  class_mode='categorical')

# test_set = test_datagen.flow_from_directory('/home/taufik/Desktop/SignLanguage/Sign-Language-to-Text/data/test',
#                                             target_size=(sz , sz),
#                                             batch_size=10,
#                                             color_mode='grayscale',
#                                             class_mode='categorical') 
# classifier.fit_generator(
#         training_set,
#         steps_per_epoch=12841, # No of images in training set
#         epochs=5,
#         validation_data=test_set,
#         validation_steps=4268)# No of images in test set


# # Saving the model
# model_json = classifier.to_json()
# with open("model-bw.json", "w") as json_file:
#     json_file.write(model_json)
# print('Model Saved')
# classifier.save_weights('model-bw.h5')
# print('Weights saved')


import cv2
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# imageSize=128
train_dir = "/home/taufik/Desktop/SignLanguage/Sign-Language-to-Text/data/train"

from tqdm import tqdm
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28           
            else:
                label = 29
            for image_filename in tqdm(os.listdir(folder +"/" +folderName)):
                img_file = cv2.imread(folder + folderName + "/" + image_filename)
                if img_file is not None:
                #     img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y
X_train, y_train = get_data(train_dir) 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2) 


