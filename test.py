from keras_preprocessing.image import load_img, img_to_array
from PIL import Image,ImageChops 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# This will work for the models if you download them from the links above. 
# If you want to export your own models, use the name of them here instead. 
model_1 = tf.keras.models.load_model('starwars_model1.h5')
# model_2 = tf.keras.models.load_model('cnn_model.h5')

def plot_image(array, i, labels):
  plt.imshow(np.squeeze(array[i]))
  plt.title(" Digit " + str(labels[i]))
  plt.xticks([])
  plt.yticks([])
  plt.show()

def predict_image(model, x):
  x = x.astype('float32')
  x = x / 255.0

  x = np.expand_dims(x, axis=0)

  image_predict = model.predict(x, verbose=0)
  label = label_as_string(np.argmax(image_predict))
  print("Predicted Label: ", label)

  plt.imshow(np.squeeze(x))
  plt.xticks([])
  plt.yticks([])
  plt.title(label)
  plt.show()
 
  return image_predict

def label_as_string(num):
  if num == 0:
    return 'Vader'
  elif num == 1:
    return 'Skywalker'
  elif num == 2:
    return 'Chewbacca'


path = "test_images/chewy_test.jpeg"
img = load_img(path, target_size=(32,32), color_mode = "rgb") 
img_arr = img_to_array(img)
arr = predict_image(model_1,  img_arr)