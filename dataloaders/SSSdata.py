import os
import torch
from torchvision.io import read_image
import torchvision.transforms.functional as TF
from torchvision.transforms import v2
import random
from torch.utils.data import Dataset


class SHIP(Dataset):
    def __init__(self, img_dir, lbl_dir, transform=True, target_transform=None, mode=None, train_files=None, val_files=None):
        """
        img_dir: path to the directory containing the images
        lbl_dir: path to the directory containing the labels
        transform: transform to be applied to the images
        target_transform: transform to be applied to the labels
        mode: "train", "test", or "val"
        train_files: list of filenames in train
        val_files: list of filenames in val
        """
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.ext = mode
        if mode == 'train':
          self.img_labels = [os.path.join(self.img_dir, train_file) for train_file in train_files]
          self.img_labels.sort()
          self.labels = [os.path.join(self.lbl_dir, train_file) for train_file in train_files]
          self.labels.sort()
        elif mode == 'val':
          self.img_labels = [os.path.join(self.img_dir, train_file) for train_file in val_files]
          self.img_labels.sort()
          self.labels = [os.path.join(self.lbl_dir, train_file) for train_file in val_files]
          self.labels.sort()
        elif mode == 'test':
          self.img_labels = list(os.listdir(os.path.join(self.img_dir, self.ext)))
          self.img_labels.sort()
          self.labels = list(os.listdir(os.path.join(self.lbl_dir, self.ext)))
          self.labels.sort()
        else:
          print('mode not valid. exiting.')
          exit()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        name = os.path.splitext(self.img_labels[idx])[0]  
        img_path = os.path.join(self.img_dir,self.ext, self.img_labels[idx])
        lbl_path = os.path.join(self.lbl_dir,self.ext, self.labels[idx])
      
        image = read_image(img_path)
        lbl = read_image(lbl_path)[0] # we only want the first channel
        
        if self.transform:
          transforms = v2.Compose([
            #v2.RandomResizedCrop(size=(224, 224), antialias=True),
            #v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.Resize(size=(512, 512)),
            v2.ToDtype(torch.float32, scale=True),
            #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])
          image = transforms(image)
        if self.target_transform:
          transforms = v2.Compose([
            #v2.RandomResizedCrop(size=(224, 224), antialias=True),
            #v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.Resize(size=(512, 512)),
            #v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])
          lbl = transforms(lbl).type(torch.LongTensor)

        #sample = {'image': image, 'gt': lbl, 'name': name}
        sample = {'image': image,
                  'label': semantic_mask,
                  'inst': instance_mask,
                  'scribble': scribble_mask}
        sample['id'] = id_
        sample['image_t'] = image_t

        if self.ext == 'train': 
            if random.random() > 0.5:
                image = TF.hflip(image)
                lbl = TF.hflip(lbl)
            if random.random() > 0.5:
                image = TF.vflip(image)
                lbl = TF.vflip(lbl)

        return sample