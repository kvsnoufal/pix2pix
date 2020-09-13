import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations
import torch
import torch.nn as nn
import models_face

# 32
# epoch=32
epoch=11
modelsavepath=os.path.join("D:/vscode repos/pix2pix/faces/log","output","models1",str(epoch),"G.pth")
G=models_face.Generator()
G.load_state_dict(torch.load(modelsavepath))
G.eval()

def order_points(pts):
  # initialzie a list of coordinates that will be ordered
  # such that the first entry in the list is the top-left,
  # the second entry is the top-right, the third is the
  # bottom-right, and the fourth is the bottom-left
  rect = np.zeros((4, 2), dtype = "float32")
  # the top-left point will have the smallest sum, whereas
  # the bottom-right point will have the largest sum
  s = pts.sum(axis = 1)
  rect[0] = pts[np.argmin(s)]
  rect[2] = pts[np.argmax(s)]
  # now, compute the difference between the points, the
  # top-right point will have the smallest difference,
  # whereas the bottom-left will have the largest difference
  diff = np.diff(pts, axis = 1)
  rect[1] = pts[np.argmin(diff)]
  rect[3] = pts[np.argmax(diff)]
  # return the ordered coordinates
  return rect
def four_point_transform(image, pts):
  # obtain a consistent order of the points and unpack them
  # individually
  rect = order_points(pts)
  (tl, tr, br, bl) = rect
  # compute the width of the new image, which will be the
  # maximum distance between bottom-right and bottom-left
  # x-coordiates or the top-right and top-left x-coordinates
  widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
  widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
  maxWidth = max(int(widthA), int(widthB))
  # compute the height of the new image, which will be the
  # maximum distance between the top-right and bottom-right
  # y-coordinates or the top-left and bottom-left y-coordinates
  heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
  heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
  maxHeight = max(int(heightA), int(heightB))
  # now that we have the dimensions of the new image, construct
  # the set of destination points to obtain a "birds eye view",
  # (i.e. top-down view) of the image, again specifying points
  # in the top-left, top-right, bottom-right, and bottom-left
  # order
  dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
  # compute the perspective transform matrix and then apply it
  M = cv2.getPerspectiveTransform(rect, dst)
  warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
  # return the warped image
  return warped    

def get_box(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (11, 11))
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    # print(len(contours))
    for cnt in contours:
        global working_cnt
        x,y,w,h = cv2.boundingRect(cnt)
        if (w>300 and h>300) and (w<400 and h<400):
            working_cnt=cnt
            # print(x,y,w,h)
            break
    rect = cv2.minAreaRect(working_cnt)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    boxed_image=cv2.drawContours(img,[box],0,(255,0,0),2)
    warped = four_point_transform(img, box)
    try:
       thresh=get_thresh(warped)
       model_input=get_model_input(thresh)
       output_tensor=G(model_input.unsqueeze(0)).squeeze(0)    
       output=convert_tensor_to_image(output_tensor)
    except:
       thresh=warped
       output=warped
    

    return boxed_image,warped,thresh,output
def get_thresh(warped):
  wcropped=warped[20:-20,20:-20]
  gray = cv2.cvtColor(wcropped.copy(), cv2.COLOR_BGR2GRAY)
  gray = cv2.blur(gray, (5, 5))
  thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
  image=(255-thresh)
  return image
aug1c=albumentations.Compose(
            [
                albumentations.Normalize(mean=(0.5), std=(0.5), max_pixel_value=1.0, always_apply=True)
            ]
        )
def get_model_input(maskedge):
  image=Image.fromarray(maskedge).convert("1")
  image=image.resize((256,256),resample=Image.BILINEAR)
  image=np.array(image).astype(np.float32)
  image=aug1c(image=image)["image"]    
  image=image.astype(np.float32)
  inp=   torch.tensor(image,dtype=torch.float).unsqueeze(0)
  return inp
def convert_tensor_to_image(tensor):
    tensor=tensor.detach().cpu()#.squeeze()
    output_image=(np.transpose(tensor.numpy().astype(np.float),(1,2,0)) + 1)/2
    # output_image=((tensor.numpy().astype(np.float)) + 1)/2
    output_image=(output_image*255).astype(np.uint8)
    # image_pil=Image.fromarray(output_image)
    output_image=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
    return output_image