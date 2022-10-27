import os
from os.path import splitext
from os import listdir
from glob import glob
import gc
import random
import logging
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from torchvision.models.segmentation import fcn_resnet101
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from scipy.ndimage import morphology

from classes import Material, BasicDataset, SegModel

import pandas as pd

import pytorch_lightning as pl

import time
NUM_GPUS = 4

inputs = {
"materials" :[
             Material("background", [85,85,85], 30, 0.5),
             Material("epidermis", [170,170,170], 150, 0.5),
             Material("mesophyll", [255,255,255], 255, 0.5),
             Material("air_space", [0,0,0], 1, 0.5),
             Material("bundle_sheath_extension", [103,103,103], 100, 0.5),
             Material("vein", (35,35,35), 180, 0.5)
            ],
#Various input/output directories
"training_image_directory" : "train/train_images/",
"training_mask_directory" : "train/train_masks/",
#Fraction of total annotations you want to leave for validating the model.
"validation_fraction": 0.2,
#Model Performance varies, make multiple models to have the best chance at success.
"num_models" : 1,
#Model Performance improves with increasing epochs, to a point.
"num_epochs" : 100,
"batch_size" : 1,
#Decrease scale to decrease VRAM usage; if you run out of VRAM during traing, restart your runtime and down scale your images
"scale" : 1,
"seed" : 0,
"models_directory" : "best_models/",
"model_group" : 'test/',
"current_model_name" : 'test',
"test_images" : "test/test_images/",
"test_masks": "test/test_masks/",
"csv_directory" : "other/",
#Input the directory of the data you want to segment here.
"inference_directory": "other/",
#Input the 5 alpha-numeric characters proceding the file number of your images
  #EX. Jmic3111_S0_GRID image_0.tif ----->mage_
"proceeding":"lice_",
#Input the 4 or mor alpha-numeric characters following the file number
  #EX. Jmic3111_S0_GRID image_0.tif ----->.tif
"following" : ".png",
"output_directory": "out/"
}


def main():
    leaf = SegModel(inputs)
    trainer = pl.Trainer(gpus = NUM_GPUS, strategy = "ddp", max_epochs = inputs['num_epochs'], progress_bar_refresh_rate=0)
    trainer.fit(leaf)
    leaf.final_validation()
    print(leaf.modeldata)
    leaf.save()
    print("Saved")
    leaf.load()
    print("Loaded")
    print(leaf.modeldata)
    
if __name__ == '__main__':
    start = time.time()
    main()
    stop = time.time()
    print("Took {:2.2f} minutes to train.".format((stop-start)/60))
