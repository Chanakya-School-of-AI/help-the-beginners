import cv2 
import numpy as np 
import time


def detect_edges(image_to_process):
    edges = cv2.Canny(image_to_process, 50, 75)
    return edges

# to read from the webcam
cap = cv2.VideoCapture(0)

# to read from a video from the hard disk
# cap = cv2.VideoCapture("stevejobs.mp4")
frame_count = 0

starttime = time.time() 
while(cap.isOpened()):

    ret, frame = cap.read()

    if ret == True:
        
        processed_frame = detect_edges(frame)

        cv2.imshow('original frame', frame)
        cv2.imshow('processed frame', processed_frame)

        frame_count = frame_count + 1   

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break 

    else:
        break
endtime = time.time()

processing_time = endtime - starttime

cap.release()
cv2.destroyAllWindows()

print("frame count: ", frame_count)
print("total process time in sec: ", processing_time)

print("fps: ", (frame_count/processing_time))