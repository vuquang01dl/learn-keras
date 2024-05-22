import cv2
import keras
from keras import layers
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (224, 224))


#cv2.imshow("img", img)
#cv2.waitKey(0)

#model
model = keras.Sequential()
model.add(layers.Conv2D(input_shape=(224, 224, 1), filters=64, kernel_size=(3, 3)))

feature_map = model.predict(np.array([img]))

#display feature map
for i in range(64):
    feature_img = feature_map[0, :, :, i]
    ax = plt.subplot(8, 8, i+1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(feature_img, cmap="gray")
plt.show()
