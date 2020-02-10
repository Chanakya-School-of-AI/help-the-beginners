import cv2
import imutils
import numpy as np 

#image = cv2.imread("4.jpg")
cap = cv2.VideoCapture(0)

while(cap.isOpened()):

    ret, image = cap.read()

    if ret == True:
        resized = imutils.resize(image, width=500)
        ratio = image.shape[0] / float(resized.shape[0])

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Thresholding! 
        #thresh = cv2.threshold(blurred, 70 , 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


        # Find contours in the images
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:

            c = c.astype("float")
            c *= ratio
            c = c.astype("int")

            cv2.drawContours(image, [c], -1, (0,255,0), 2)
            cv2.imshow("final", image)
            cv2.waitKey(1)
    else:
        break

cap.release()
cv2.destroyAllWindows()