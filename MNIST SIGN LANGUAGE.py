# -*- coding: utf-8 -*-
"""
Created on Sat May  4 23:35:18 2019

@author: Josh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping

train_load = pd.read_csv('sign_mnist_train.csv')

y_train = train_load['label']

#Change dtype to category
y_train = y_train.astype('category')

#Get dummies
y_train = to_categorical(y_train)

X_train = train_load

X_train = X_train.drop('label',axis=1)

X_train = X_train / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

X_train_imgdat = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_center=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    zoom_range=.05,
                                    rotation_range=5,
                                    height_shift_range=.05,
                                    width_shift_range=.05,)

X_train_imgdat.fit(X_train)

model = Sequential()

#Conv input layer
model.add(Conv2D(32,
                 kernel_size=3,
                 activation='relu',
                 input_shape=(28,28,1)))

#Second Conv layer
model.add(Conv2D(filters = 32,
                 kernel_size = (3,3),
                 padding = 'same', 
                 activation ='relu'))

#Possibly add a normalization layer, taken
#model.add(BatchNormalization())

#Pool pixels into a square 2X2 pool
model.add(MaxPool2D(pool_size=(2,2)))

#Add a .3 dropout to prevent overfitting
model.add(Dropout(0.3))

#Conv layer after pool and dropout
model.add(Conv2D(filters = 64,
                 kernel_size = (3,3),
                 padding = 'same', 
                 activation ='relu'))

#Flatten data to be classified
model.add(Flatten())

#Classify layer
model.add(Dense(25,activation='softmax'))

#Compile using the adam optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#To make saving checkpoints
checkpoint = ModelCheckpoint('mnist-checkpoint',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             mode='auto',
                             period=1)

#To possibly stop early the epochs
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=5,
                          verbose=1,
                          mode='auto')

fit = model.fit_generator(X_train_imgdat.flow(X_train,y_train,batch_size=32),
                    epochs=35,
                    verbose=2,
                    steps_per_epoch=X_train.shape[0] // 32,
                    callbacks=[checkpoint,earlystop])

test_load = pd.read_csv('sign_mnist_test.csv')

test_label = test_load['label']
test_label = test_label.astype('category')
test_label = to_categorical(test_label)

test_load = test_load.drop('label',axis=1)
test_load = test_load / 255

test = test_load.values.reshape(-1,28,28,1)

model.evaluate(test,test_label)
