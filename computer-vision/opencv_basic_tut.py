import cv2

img = cv2.imread("Dog.jpg")

# Note - opencv reads the image in BGR format! 
# But all the deep learning frameworks expect the image to be in RGB.

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imshow('rgb', rgb_img)
cv2.imshow('frame', img)
cv2.waitKey(0) # press shift to close the image


