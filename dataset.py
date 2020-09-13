from PIL import Image
import numpy as np
import torch
import albumentations
import random
import os
import cv2
import img_utils
import config


class ColorizeDataset():
    def __init__(self,filepaths,test=False):
        self.files=filepaths
        
        self.aug=albumentations.Compose(
            [
                albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=True)
            ]
        )
       
    def __len__(self):
        return len(self.files)
    def __getitem__(self,item):
        image=Image.open(self.files[item])
        image=image.convert("RGB")
        image=image.resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH),resample=Image.BILINEAR)

        target_image=image
        image=image.convert("1").convert("RGB")

        image=np.array(image)
        image=self.aug(image=image)["image"]

        target_image=np.array(target_image)
        target_image=self.aug(image=target_image)["image"]

        if random.random()<0.5:
            image=image[:,::-1,:]
            target_image=target_image[:,::-1,:]


        image=np.transpose(image,(2,0,1)).astype(np.float32)
        target_image=np.transpose(target_image,(2,0,1)).astype(np.float32)

        return {
            "image":torch.tensor(image,dtype=torch.float),
            "target_image":torch.tensor(target_image,dtype=torch.float)
            }

class InfillDatasetCutout():
    def __init__(self,filepaths,test=False):
        self.files=filepaths
        
        self.aug=albumentations.Compose(
            [
                albumentations.Cutout(
                                    num_holes=10,
                                    max_h_size=20,
                                    max_w_size=20,
                                    always_apply=True,
                                    fill_value=255
                                    
                                ),
                albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=True)
            ]
        )
        self.target_aug=albumentations.Compose(
            [
                albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=True)
            ]
        )


    def __len__(self):
        return len(self.files)
    def __getitem__(self,item):
        image=Image.open(self.files[item])
        image=image.convert("RGB")
        image=image.resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH),resample=Image.BILINEAR)

        target_image=image
        

        image=np.array(image)
        image=image*img_utils.random_mask(config.IMAGE_HEIGHT,config.IMAGE_WIDTH)
        image=self.aug(image=image)["image"]

        target_image=np.array(target_image)
        target_image=self.target_aug(image=target_image)["image"]

        if random.random()<0.5:
            image=image[:,::-1,:]
            target_image=target_image[:,::-1,:]


        image=np.transpose(image,(2,0,1)).astype(np.float32)
        target_image=np.transpose(target_image,(2,0,1)).astype(np.float32)

        return {
            "image":torch.tensor(image,dtype=torch.float),
            "target_image":torch.tensor(target_image,dtype=torch.float)
            }            


class ABDataset():
    def __init__(self,filepaths1,filepaths2,test=False):
        self.img_files=filepaths1
        # self.lbl_files=filepaths2
        self.lbl_files=[os.path.join(filepaths2,os.path.basename(t)) for t in filepaths1]
        self.aug3c=albumentations.Compose(
            [
                albumentations.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=True)
            ]
        )
        self.aug1c=albumentations.Compose(
            [
                albumentations.Normalize(mean=(0.5), std=(0.5), max_pixel_value=1.0, always_apply=True)
            ]
        )
       


    def __len__(self):
        return len(self.img_files)
    def __getitem__(self,item):
        image=Image.open(self.lbl_files[item])
        target_image=Image.open(self.img_files[item])
        image=image.convert("1")
        image=image.resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH),resample=Image.BILINEAR)
        target_image=target_image.convert("RGB")
        target_image=target_image.resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH),resample=Image.BILINEAR)

        

        image=np.array(image).astype(np.float32)
        image=self.aug1c(image=image)["image"]

        target_image=np.array(target_image)
        target_image=self.aug3c(image=target_image)["image"]

        if random.random()<0.5:
            image=image[:,::-1]
            target_image=target_image[:,::-1,:]


        image=image.astype(np.float32)
        target_image=np.transpose(target_image,(2,0,1)).astype(np.float32)

        return {
            "image":torch.tensor(image,dtype=torch.float).unsqueeze(0),
            "target_image":torch.tensor(target_image,dtype=torch.float)
            }