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

import pandas as pd

import pytorch_lightning as pl



class Material:
 
  def __init__(self, name, input_rgb_vals, output_val, confidence_threshold=0):
    self.name = name
    self.input_rgb_vals = input_rgb_vals
    self.output_val = output_val
    self.confidence_threshold = confidence_threshold

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, materials, scale=1, transform=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transform=transform
        self.t_list=A.Compose([A.HorizontalFlip(p=0.4),A.VerticalFlip(p=0.4), A.Rotate(limit=(-50, 50), p=0.4),])
        self.means=[0,0,0]
        self.stds=[1,1,1]
        self.materials = materials
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
 
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
 
    def __len__(self):
        return len(self.ids)
 
 
    @classmethod
    def mask_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
 
        img_nd = np.array(pil_img)
 
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
       
        return img_nd
    
 
    def img_preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
 
        img_nd = np.array(pil_img)
 
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        return img_nd
 
    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
 
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])
 
        #Reshapes from 1 channel to 3 channels in grayscale
        img = self.img_preprocess(img, self.scale)
        mask = self.mask_preprocess(mask, self.scale)
        new_image=np.zeros((img.shape[0],img.shape[1],3))
        new_image[:,:,0]=img[:,:,0]
        new_image[:,:,1]=img[:,:,0]
        new_image[:,:,2]=img[:,:,0]
        
        img=new_image
 
        #New Code
        masklist=[]
            
        for i, mat in enumerate(self.materials):
        
          indices = np.all(mask == mat.input_rgb_vals, axis=-1)
          new_mask=np.zeros((img.shape[0],img.shape[1]))
          new_mask[indices] = 1
          masklist.append(new_mask)
 
        mask=masklist
        
        if img.max() > 1:
            img = img / 255
 
        if self.transform:
            augmented=self.t_list(image=img, masks=mask)
            img=augmented["image"]
            mask=augmented["masks"]
            
        img = img.transpose((2, 0, 1))
        
        mask=np.array(mask)
        
        img=torch.from_numpy(img)
        mask=torch.from_numpy(mask)
        
        img=transforms.Normalize(mean=self.means, std=self.stds)(img)
        return img, mask
    
class SegModel(pl.LightningModule):
    def __init__(self, init_dict):
        super().__init__()
        self.__dict__.update(init_dict)
        self.num_materials = len(init_dict["materials"])
        self.dir_checkpoint = self.models_directory
        os.makedirs(self.dir_checkpoint+self.model_group, exist_ok=True)
        
        ## gets all data ready
        self.dataset = self.setup_data()
        self.dataset_train, self.dataset_val = self.trainval_split(self.dataset)
        self.train_loader = self.get_loader(self.dataset_train)
        self.val_loader = self.get_loader(self.dataset_val)
        
        #model set up
        self.model = fcn_resnet101(pretrained=True, progress=True)
        self.model.classifier=FCNHead(2048, self.num_materials)
        self.criterion = nn.BCEWithLogitsLoss()
        
    def save(self, PATH = self.model.state_dict(), self.dir_checkpoint + self.model_group + self.current_model_name+".out"):
        torch.save(PATH)
        
    def load(self, PATH = self.dir_checkpoint + self.model_group + self.current_model_name+".out"):
        self.model.load_state_dict(torch.load(PATH))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return [optimizer]
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img.float())
        loss = self.criterion(pred["out"], mask)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img.float())
        loss = self.criterion(pred["out"], mask)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def get_data(self):
        return BasicDataset(self.training_image_directory, self.training_mask_directory, self.materials, scale=self.scale, transform=False)
    
    def get_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    def trainval_split(self, dataset):
        validation_size = int(len(dataset) * self.validation_fraction)
        train_size = len(dataset) - validation_size
        train, val = torch.utils.data.random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(self.seed))
        return train, val

    def setup_data(self):
        dataset = self.get_data()
        train_loader = self.get_loader(dataset)
        nimages = 0
        mean = 0. 
        std = 0.
        for batch, _ in train_loader:
            # Rearrange batch to be the shape of [B, C, W * H]
            batch = batch.view(batch.size(0), batch.size(1), -1)
            # Update total number of images
            nimages += batch.size(0)
            # Compute mean and std here
            mean += batch.mean(2).sum(0) 
            std += batch.std(2).sum(0)

        # Final step
        mean /= nimages
        std /= nimages

        dataset.means=mean
        dataset.stds=std 
        
        return dataset
    
    def train_dataloader(self):
        return self.get_loader(self.dataset_train)

    def val_dataloader(self):
        return self.get_loader(self.dataset_val)

    def test_dataloader(self):
        return self.get_loader(self.dataset_val)

    def test_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img.float())
        loss = self.criterion(pred, mask)
        return loss
    
    def predict_step(self, batch, batch_idx):
        img, mask = batch
        pred = self.forward(img.float())
        return pred["out"]
    
    def final_validation(self): 
        prop_list = []
        for mat in self.materials:
                prop_list.append([[],[],[],[]])

        for images, target in self.val_loader:
            images = images.float()
            target = target.float()
            with torch.no_grad():
                pred=self.model(images)['out']
                pred=nn.Sigmoid()(pred)

            for i, mat in enumerate(self.materials):
                material_target = target[:,i,:,:]
                material_pred = pred[:, i, :, :]
                material_pred[material_pred >=mat.confidence_threshold] = 1
                material_pred[material_pred <=mat.confidence_threshold] = 0
                pred[:, i, :, :]=material_pred

                material_tp=torch.sum(material_target*material_pred, (1,2))
                material_fp=torch.sum((1-material_target)*material_pred, (1,2))
                material_fn=torch.sum(material_target*(1-material_pred), (1,2))
                material_tn=torch.sum((1-material_target)*(1-material_pred), (1,2))

                material_precision=torch.mean((material_tp+0.000000001)/(material_tp+material_fp+0.000000001))
                material_recall=torch.mean((material_tp+0.000000001)/(material_tp+material_fn+0.000000001))
                material_accuracy=torch.mean((material_tp+material_tn+0.000000001)/(material_tp+material_tn+material_fp+material_fn+0.000000001))
                material_f1=torch.mean(((material_tp+0.000000001))/(material_tp++0.000000001+0.5*(material_fp+material_fn)))

                prop_list[i][0].append(material_precision.cpu().detach().numpy())
                prop_list[i][1].append(material_recall.cpu().detach().numpy())
                prop_list[i][2].append(material_accuracy.cpu().detach().numpy())
                prop_list[i][3].append(material_f1.cpu().detach().numpy())

        properties = {"name" : [mat.name for mat in self.materials],
                "precision" : [str(np.mean(prop_list[i][0])) for i in range(self.num_materials)],
                "recall" : [str(np.mean(prop_list[i][1])) for i in range(self.num_materials)],
                "accuracy" : [str(np.mean(prop_list[i][2])) for i in range(self.num_materials)],
                "f1" : [str(np.mean(prop_list[i][3])) for i in range(self.num_materials)]}
        self.modeldata = pd.DataFrame(properties, columns = ["name", "precision", "recall", "accuracy", "f1"])

        
        
        
class SegModelRegular():
    def __init__(self, init_dict):
        self.__dict__ = init_dict
        self.num_materials = len(init_dict["materials"])
        self.dir_checkpoint = self.models_directory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.dir_checkpoint+self.model_group, exist_ok=True )
        
    def set_up_model(self):
        self.model = fcn_resnet101(pretrained=True, progress=True)
        self.model.classifier=FCNHead(2048, self.num_materials)
        return self.model
    
    def get_data(self):
        return BasicDataset(self.training_image_directory, self.training_mask_directory, self.materials, scale=self.scale, transform=False)
    
    def get_loader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    def trainval_split(self, dataset):
        validation_size = int(len(dataset) * self.validation_fraction)
        train_size = len(dataset) - validation_size
        train, val = torch.utils.data.random_split(dataset, [train_size, validation_size], generator=torch.Generator().manual_seed(self.seed))
 
        return train, val

    def setup_data(self):
        dataset = self.get_data()
        train_loader = self.get_loader(dataset)
        nimages = 0
        mean = 0. 
        std = 0.
        for batch, _ in train_loader:
            # Rearrange batch to be the shape of [B, C, W * H]
            batch = batch.view(batch.size(0), batch.size(1), -1)
            # Update total number of images
            nimages += batch.size(0)
            # Compute mean and std here
            mean += batch.mean(2).sum(0) 
            std += batch.std(2).sum(0)

        # Final step
        mean /= nimages
        std /= nimages

        print(mean)
        print(std)

        dataset.means=mean
        dataset.stds=std 
        
        return dataset
    
    def train(self,verbose = True):
        self.model = self.set_up_model() 
        self.dataset = self.get_data()
        self.dataset_train, self.dataset_val = self.trainval_split(self.dataset)

        self.train_loader = self.get_loader(self.dataset_train)
        self.val_loader = self.get_loader(self.dataset_val)

        self.model.to(self.device)

        num_epochs= self.num_epochs
        optimizer=torch.optim.Adam(self.model.parameters(), lr=1e-3)

        best_loss=999

        criterion = nn.BCEWithLogitsLoss()

        #this is the train loop
        for epoch in range(num_epochs):
            #print(psutil.virtual_memory().percent)
            if verbose:
                print('Epoch: ', str(epoch))
        #add back if doing fractional training
            self.train_loader.dataset.dataset.transform=True
            self.model.train()
            for images, masks in self.train_loader:

                images = images.to(device=self.device, dtype=torch.float32)
                masks = masks.to(device=self.device, dtype=torch.float32)

                #forward pass
                preds=self.model(images)['out'].cuda()

                #compute loss
                loss=criterion(preds, masks)

                #reset the optimizer gradients to 0
                optimizer.zero_grad()

                #backward pass (compute gradients)
                loss.backward()

                #use the computed gradients to update model weights
                optimizer.step()

            if verbose:
                print('Train loss: '+str(loss.to('cpu').detach()))

        self.val_loader.dataset.dataset.transform=False
        current_loss=0

        #test on val set and save the best checkpoint
        self.model.eval()
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(device=self.device, dtype=torch.float32)
                masks = masks.to(device=self.device, dtype=torch.float32)
                preds = self.model(images)['out'].cuda()

                loss = criterion(preds, masks)
                current_loss+=loss.to('cpu').detach()
                del images, masks, preds, loss
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!Re-name model here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        if best_loss>current_loss:
            best_loss=current_loss
            print('Best Model Saved!, loss: '+ str(best_loss))
            torch.save(self.model.state_dict(), self.dir_checkpoint + self.model_group + self.current_model_name+".pth")
        else:
            print('Model is bad!, Current loss: '+ str(current_loss) + ' Best loss: '+str(best_loss))
            print('\n')
    def validation(self): 
        prop_list = []
        for mat in self.materials:
            prop_list.append([[],[],[],[]])

        for images, target in self.val_loader:
            images = images.to(device=self.device, dtype=torch.float32)
            target = target.to(device=self.device, dtype=torch.float32)

            with torch.no_grad():
                pred=self.model(images)['out'].cuda()
                pred=nn.Sigmoid()(pred)

            for i, mat in enumerate(self.materials):
                material_target = target[:,i,:,:]
                material_pred = pred[:, i, :, :]
                material_pred[material_pred >=mat.confidence_threshold] = 1
                material_pred[material_pred <=mat.confidence_threshold] = 0
                pred[:, i, :, :]=material_pred

                material_tp=torch.sum(material_target*material_pred, (1,2))
                material_fp=torch.sum((1-material_target)*material_pred, (1,2))
                material_fn=torch.sum(material_target*(1-material_pred), (1,2))
                material_tn=torch.sum((1-material_target)*(1-material_pred), (1,2))

                material_precision=torch.mean((material_tp+0.000000001)/(material_tp+material_fp+0.000000001))
                material_recall=torch.mean((material_tp+0.000000001)/(material_tp+material_fn+0.000000001))
                material_accuracy=torch.mean((material_tp+material_tn+0.000000001)/(material_tp+material_tn+material_fp+material_fn+0.000000001))
                material_f1=torch.mean(((material_tp+0.000000001))/(material_tp++0.000000001+0.5*(material_fp+material_fn)))

                prop_list[i][0].append(material_precision.cpu().detach().numpy())
                prop_list[i][1].append(material_recall.cpu().detach().numpy())
                prop_list[i][2].append(material_accuracy.cpu().detach().numpy())
                prop_list[i][3].append(material_f1.cpu().detach().numpy())

        properties = {"name" : [mat.name for mat in self.materials],
                "precision" : [str(np.mean(prop_list[i][0])) for i in range(self.num_materials)],
                "recall" : [str(np.mean(prop_list[i][1])) for i in range(self.num_materials)],
                "accuracy" : [str(np.mean(prop_list[i][2])) for i in range(self.num_materials)],
                "f1" : [str(np.mean(prop_list[i][3])) for i in range(self.num_materials)]}
        self.modeldata = pd.DataFrame(properties, columns = ["name", "precision", "recall", "accuracy", "f1"])


