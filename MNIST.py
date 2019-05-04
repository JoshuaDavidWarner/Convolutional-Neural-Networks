# -*- coding: utf-8 -*-
"""
Created on Wed May  1 01:30:56 2019

@author: Josh
"""
#Load Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,EarlyStopping

#Load in training and test data
train_load = pd.read_csv('train.csv')

test_load = pd.read_csv('test.csv')

#Drop the target varaible
X_train = train_load.drop('label',
                          axis=1)
#Create a target variable
y_train = train_load['label']

#Change dtype to category
y_train = y_train.astype('category')

#Get dummies
y_train = to_categorical(y_train)

#Change color scale to gray scale
X_train = X_train / 255.0

#Reshape data to original data dimensions of 28X28
X_train = X_train.values.reshape(-1,28,28,1)

#Create more samples by manipulating the samples
X_train_imgdat = ImageDataGenerator(featurewise_center=False,
                                    featurewise_std_normalization=False,
                                    samplewise_center=False,
                                    samplewise_std_normalization=False,
                                    zca_whitening=False,
                                    zoom_range=.15,
                                    rotation_range=15,
                                    height_shift_range=.05,
                                    width_shift_range=.05,)

#Fit data to generator
X_train_imgdat.fit(X_train)

#Instantiate model object
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
model.add(Dense(10, 
                activation='softmax'))

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

#Fit generator to the model
fit = model.fit_generator(X_train_imgdat.flow(X_train,y_train,batch_size=32),
                    epochs=35,
                    verbose=2,
                    steps_per_epoch=X_train.shape[0] // 32,
                    callbacks=[checkpoint,earlystop])

#Change the test data to a grayscale
test_load = test_load / 255

#Reshape test values
test = test_load.values.reshape(-1,28,28,1)

#Predict using test values
y_preds = model.predict(test)

#Round to a classification
y_preds = np.round(y_preds)

#Combine labels to 1 stack
y_preds = pd.DataFrame(y_preds).idxmax(axis=1)

#Save the weights
model.save_weights('mnist-1.h5')

#A df for submission
submission = pd.DataFrame(list(range(1,28001)))

#label column appropriately
submission['ImageId'] = submission[0]

#Change order
submission = submission.drop(0,axis=1)

#label column appropriately
submission['Label'] = y_preds

#Submit the predictions to a csv
submission.to_csv('submission.csv',index=False)

#Check the history
history = fit.history

#Graphing the history
plt.plot(history['loss'])

plt.plot(history['acc'])

plt.show()
