

import cv2
import sys
import os

VIDEO_TO_CONVERT=sys.argv[1]
DESTINATION_FOLDER=sys.argv[2]
# Opens the Video file
cap= cv2.VideoCapture(VIDEO_TO_CONVERT)
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite(os.path.join(DESTINATION_FOLDER,str(i)+".jpg"),frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()