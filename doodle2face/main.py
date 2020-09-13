PROJECT="doodle2face"
import sys
sys.path.insert(0,"./")
import torch
import torch.nn as nn
import torchvision
from glob import glob
import random
import numpy as np
import models
from tqdm import tqdm
import os
from PIL import Image
import dataset
import engine
import config
import pandas as pd

torch.cuda.empty_cache()


if __name__=="__main__":
    train_path_images=config.DOODLE2FACE["TRAINPATH_IMAGES"]
    train_path_targets=config.DOODLE2FACE["TRAINPATH_TARGETS"]
    train_files=glob(os.path.join(train_path,"*.jpg"))
    print(f"TRAIN FILES LOADED : {len(train_files)}")

    test_path=config.DOODLE2FACE["TESTPATH"]
    test_files=glob(os.path.join(test_path,"*.jpg"))
    print(f"TEST FILES LOADED: {len(test_files)}")

    train_dataset=dataset.ABDataset(train_path_images,train_path_targets)
    test_dataset=dataset.ABDataset(test_files,train_path_targets)

    train_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=12
    )
    test_loader=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    G=models.GeneratorOneChannelInput().to(config.DEVICE)
    D=models.DiscriminatorOneChannelInput().to(config.DEVICE)

    G.apply(models.init_weights)
    D.apply(models.init_weights)

    oG=torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    oD=torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    criterion=nn.BCEWithLogitsLoss().to(config.DEVICE)
    l1loss=nn.L1Loss().to(config.DEVICE)

    E=engine.Engine(PROJECT,G,D,criterion,l1loss,oD,oG,config.DEVICE)
    losses=[]
    for epoch in range(config.EPOCHS):
        train_loss=E.train(train_loader,epoch)
        print(epoch,train_loss)
        E.generate(test_loader,epoch)
        
        losses.append(train_loss)
        
        # break

    logdf=pd.DataFrame(losses).reset_index()

    logdf.columns=["epochs","GLoss","Dloss"]
    logdf.to_csv(os.path.join(PROJECT,config.LOGPATH,"log_df.csv"))


