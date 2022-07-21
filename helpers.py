from PIL import Image
import os
import numpy as np 
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten


def images_to_df(search_term, new_frame, new_size, image_path, label_encode):

    label_frame = np.empty((0,), dtype=int)
    path = '/'.join((image_path, search_term))

    for image in os.listdir(path):
    
        full_path = '/'.join((path, image))

        if image == '.DS_Store':
            pass
        else:
            image = Image.open(full_path)
            image = image.resize((new_size, new_size))
            arr = np.asarray(image)
            arr = arr.reshape(1, new_size, new_size, 3)
            new_frame = np.append(new_frame, arr, axis=0)

            encode = np.array([label_encode])
            label_frame = np.append(label_frame, encode, axis = 0)


    return new_frame, label_frame 

def get_model(img_rows, img_cols, num_pixels):
    model = Sequential()
    input_shape = (img_rows, img_cols, num_pixels)

    model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    return model 
