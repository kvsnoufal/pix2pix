import torch
import torch.nn as nn
import config
from tqdm import tqdm
import numpy as np
import torchvision
import os
import img_utils

class Engine:
    def __init__(self,project,G,D,criterion,l1loss,oD,oG,device):
        self.project=project
        self.G=G
        self.D=D
        self.criterion=criterion
        self.l1loss=l1loss
        self.oD=oD
        self.oG=oG
        self.device=device
    def gen_loss(self,disc_output,gen_output,target):
        gan_loss=self.criterion(disc_output,torch.ones_like(disc_output))

        l1_loss=self.l1loss(gen_output,target)

        g_loss=gan_loss + (config.LAMBDA* l1_loss)
        return g_loss
    def disc_loss(self,disc_real,disc_fake):
        real_loss=self.criterion(disc_real,torch.ones_like(disc_real))
        fake_loss=self.criterion(disc_fake,torch.zeros_like(disc_fake))
        d_loss=real_loss+fake_loss
        return d_loss

    def train(self,dataloader,epoch):
        self.G.train()
        self.D.train()
        g_losses=[]
        d_losses=[]
        for indx,data in tqdm(enumerate(dataloader)):
            self.D.zero_grad()
            image=data["image"].to(self.device)
            target=data["target_image"].to(self.device)

            real_disc_output=self.D(image,target).squeeze()
            fake=self.G(image)
            fake_disc_output=self.D(fake.detach(),target).squeeze()
            d_loss=self.disc_loss(real_disc_output,fake_disc_output)
            d_loss.backward()
            self.oD.step()

            self.G.zero_grad()
            disc_output=self.D(fake,target).squeeze()
            g_loss=self.gen_loss(disc_output,fake,target)
            g_loss.backward()
            self.oG.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        return np.mean(g_losses),np.mean(d_losses)
    
    def validate(self,dataloader):

        pass
    def generate(self,dataloader,epoch,savemodel=True):
        with torch.no_grad():
            
            os.makedirs(os.path.join(self.project,config.LOGPATH,str(epoch)), exist_ok=True)
            imagesavefolder=os.path.join(self.project,config.LOGPATH,str(epoch))
            if savemodel:
                os.makedirs(os.path.join(self.project,config.LOGPATH,"models",str(epoch)), exist_ok=True)
                modelsavepath=os.path.join(self.project,config.LOGPATH,"models",str(epoch),"G.pth")
                torch.save(self.G.state_dict(), modelsavepath)
            self.G.eval()
            self.D.eval()
            for indx,data in enumerate(dataloader):
                image,target_image=data["image"].cuda(),data["target_image"].cuda()
                fake=self.G(image)

                

                img_utils.save_image_from_dataloader3c(image,imagesavefolder,"inp",indx)
                img_utils.save_image_from_dataloader3c(fake,imagesavefolder,"output",indx)
                img_utils.save_image_from_dataloader3c(target_image,imagesavefolder,"gt",indx)
               
               
    


    
