# author : Vipul Vaibhaw
# inspired - https://stackoverflow.com/questions/33227202/detecting-scratch-on-image-with-much-noise

import cv2
import numpy as np

img = cv2.imread("./dataset/Kn9wT.jpg")

# resizing image for faster calculations
img = cv2.resize(img, (int(0.5*img.shape[1]), int(0.5*img.shape[0])))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray,(15,15),0)

scale = 1
delta = 0
ddepth = cv2.CV_16S
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)


abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)


grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

edges = cv2.Canny(grad,30,100)

# morpholgical operation Dilation followed by Erosion of grayscale image by a mask of 3 X 3.
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_show = cv2.drawContours(img , contours, -1, (0, 255, 0), 3)
cv2.imshow('result', closing) 
cv2.waitKey(0)