import cv2
from tensorflow import keras
#from keras.layers.core import Dense
from keras import layers


img = cv2.imread("red_panda_2.jpg", cv2.IMREAD_GRAYSCALE)

height, width = img.shape
print(img.shape)

#keras model structrure
input_layer = keras.Input(shape=(height, width))

print("Input layer shape: ", input_layer.shape)
layer_1 = layers.Dense(64)(input_layer)
layer_2 = layers.Dense(32)(layer_1)
output = layers.Dense(2)(layer_2)


#define model

model =  keras.Model(inputs = input_layer, outputs = output)
model.summary()

cv2.imshow("red panda", img)
cv2.waitKey(0)