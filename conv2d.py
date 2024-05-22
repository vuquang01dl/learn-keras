import cv2
from tensorflow import keras
from keras import layers

#load img
img = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
#img = cv2.resize(img, (250, 250))
height, width = img.shape

cv2.imshow("img", img)

model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(height, width, 1), filters=32, kernel_size=(5, 5)))

model.summary()

#access layers parameters
filters, _ = model.layers[0].get_weights()
f_min, f_max = filters.min(), filters.max()

print(f_min, f_max)

f = filters[:,:,:,0]
f = cv2.resize(f, (height, width), interpolation=cv2.INTER_NEAREST)
cv2.imshow("filter", f)
cv2.waitKey(0)