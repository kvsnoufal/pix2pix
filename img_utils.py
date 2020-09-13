from numpy.random import randint
import numpy as np
import cv2
import torchvision
import torch
import torch.nn as nn
from PIL import Image
import os
def random_mask(height, width, channels = 3):
    img = np.zeros((height, width, channels), np.uint8)
    
    # Set scale
    size = int((width + height) * 0.007)
    if width < 64 or height < 64:
        raise Exception("Width and Height of maks must be at least 64")
    
    # Draw Random Lines
    for _ in range(randint(1, 20)):
        x1, x2 = randint(1, width), randint(1, width)
        y1, y2 = randint(1, height), randint(1, height)
        thickness = randint(1, size)
        cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)
    
    # Draw Random Circles
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(1, size)
        cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)
    
    # Draw Random Ellipses
    for _ in range(randint(1, 20)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(1, 180), randint(1, 180), randint(1, 180)
        thickness = randint(1, size)
        cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

    return 1 - img     
def save_image_from_dataloader3c(image,imagesavefolder,prefix,indx):
    image=image.cpu()
    image = torchvision.utils.make_grid(image)
    image=(np.transpose(image.numpy().astype(np.float),(1,2,0))+1)/2
    image=(image*255).astype(np.uint8)
    image_pil=Image.fromarray(image)
    image_pil.save(os.path.join(imagesavefolder,f"{prefix}_{indx}.jpg"))    
    pass