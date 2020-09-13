from PIL import ImageGrab
import numpy as np
import cv2
from img_utils import get_box

while True:
    img = ImageGrab.grab(bbox=(0,160,640,640)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
    img_np = np.array(img) #this is the array obtained from conversion
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    boxed_image,warped,threshd,modeloutput=get_box(frame)
    cv2.imshow("input stream - boxed", boxed_image)
    cv2.imshow("warped extract", warped)
    cv2.imshow("model input", threshd)
    cv2.imshow("GAN output", modeloutput)
    if cv2.waitKey(1) == 27:
        exit(0)

        cv2.destroyAllWindows()
