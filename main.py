import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from PIL import Image
from helpers import images_to_df, get_model
from tensorflow import keras
from sklearn.model_selection import train_test_split

## read in CSV files of image names and specify where they are saved ##
image_path = '/Users/colinmcgravey/Documents/WebScraper/augmented'

## specify new size of image in dataframe, will maintain aspect ratio ##
img_rows, img_cols = 32, 32
num_classes = 2

## converts the Vader images into an array ##
arr = np.empty((0, img_rows, img_cols, 3), dtype=int)
Vader_images, Vader_labels = images_to_df('vader', arr, img_rows, image_path, 0)

## converts the Skywalker images into an array ##
arr2 = np.empty((0, img_rows, img_cols, 3), dtype=int)
Skywalker_images, Skywalker_labels = images_to_df('anakin', arr2, img_rows, image_path, 1)

print(Vader_images.shape)
print(Skywalker_images.shape)

print(Vader_labels.shape)
print(Skywalker_labels.shape)

## creates the dataset and labels for train and testing ##
Dataset = np.append(Vader_images, Skywalker_images, axis = 0)
Labels = np.append(Vader_labels, Skywalker_labels, axis = 0)

## normalizes pixel values within the dataset to be between 0 and 1 ##
Dataset = Dataset.astype('float32')
Dataset /= 255

## one hot encode the labels for categorical data ## 
Labels = keras.utils.to_categorical(Labels, num_classes)

## split the data into training and testing using train_test_split ## 
X_train, X_test, Y_train, Y_test = train_test_split(Dataset, Labels, test_size=0.05, random_state=4)

## retrieve model from the get_model() function ##
model = get_model(img_rows, img_cols, 3)

## compiles and trains model ## 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, shuffle=True)

## saves model ## 
model.save('starwars_model.h5')