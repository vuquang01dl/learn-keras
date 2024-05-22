import cv2

img = cv2.imread("redpanda.jpg")
img2 = cv2.resize(img, (224, 224))
cv2.imshow("anh", img)
cv2.imshow("anh2", img2)
cv2.waitKey(0)