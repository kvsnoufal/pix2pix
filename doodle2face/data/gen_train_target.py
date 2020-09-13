
import sys
sys.path.insert(0,"./")
import cv2 
from PIL import Image
import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import numpy as np
# import argparse
import config



import dlib
import cv2

files=glob("source_files/*.jpg")
global net
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def preprocess_cv(img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        inp = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(256, 256),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
        net.setInput(inp)
        out = net.forward()
        out = out[0, 0]
        out = cv2.resize(out, (256,256))
        out = 255 * out
        out = out.astype(np.uint8)
        out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)
        return img,out

cv2.dnn_registerLayer('Crop', CropLayer)
net = cv2.dnn.readNet(config.HOLY_MODEL_CONFIG, config.HOLY_MODEL)
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config.FACELANDMARK_MODEL)   
indx=0 
for file in files:
    # print(file)
    image=cv2.imread(file)
    image=cv2.resize(image,(256,256), interpolation = cv2.INTER_CUBIC)
    mask=np.zeros(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # print(len(rects))
    if(len(rects)==0):
        continue
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        
        for d,i in enumerate(range(16)):
            if(i==len(shape)-1):
                continue

            cv2.line(mask,(shape[i][0],shape[i][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        for d,i in enumerate(range(17,21)):
            # print(i)
            cv2.line(mask,(shape[i][0],shape[i][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        for d,i in enumerate(range(22,26)):
            # print(i)
            cv2.line(mask,(shape[i][0],shape[i][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        for d,i in enumerate(range(27,30)):
            # print(i)
            cv2.line(mask,(shape[i][0],shape[i][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        for d,i in enumerate(range(31,35)):
            # print(i)
            cv2.line(mask,(shape[i][0],shape[i][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        for d,i in enumerate(range(36,41)):
            # print(i)
            cv2.line(mask,(shape[i][0],shape[i][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        cv2.line(mask,(shape[36][0],shape[36][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        for d,i in enumerate(range(42,47)):
            # print(i)
            cv2.line(mask,(shape[i][0],shape[i][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        cv2.line(mask,(shape[42][0],shape[42][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        for d,i in enumerate(range(48,60)):
            # print(i)
            cv2.line(mask,(shape[i][0],shape[i][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        cv2.line(mask,(shape[48][0],shape[48][1]),(shape[i+1][0],shape[i+1][1]),(255,255,255),5)
        break
    
    
    target,maskedge=preprocess_cv(image)
    y_thresh=shape[30][1]

    # maskedge[y_thresh:,:,:]=0
    # maskedge[np.where(maskedge>200),255,0]
    maskedgethres=np.where(maskedge>100,255,0)
    maskedgethres=(maskedgethres+mask).clip(0,255)

    # print(image.shape)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=np.array(image)
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_image=np.array(maskedgethres)
    image_pil=Image.fromarray(image)#.convert("RGB")
    image_pil.save(os.path.join("doodle2face/data/train",f"{indx}.jpg"))
    # print(target_image.shape)
    # target_image=cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    image_pil=Image.fromarray(target_image.astype(np.uint8))#.convert("RGB")
    image_pil.save(os.path.join("doodle2face/data/targets/",f"{indx}.jpg"))
    indx+=1
    
    
