import cv2
from tensorflow import keras
from keras import layers
import numpy as np


img = cv2.imread("red_panda_2.jpg")

height, width, channels = img.shape
print("height: {}, witdth: {}, channels: {}".format(height, width, channels))

#create  sequential model
model = keras.Sequential()
model.add(layers.Input(shape=(height, width, channels)))
model.add(layers.Dense(32))
model.add(layers.Dense(16))
model.add(layers.Dense(2))

preprocessed_img = np.array([img])

result = model(preprocessed_img)
print(result)

