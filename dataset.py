import torch
import torchvision
#import torchxrayvision as xrv
import torchvision, torchvision.transforms
import skimage
from skimage.io import imread, imsave
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import numpy as np
import skimage.transform as scikit_transform
from sklearn.utils import shuffle
from skimage import color
from skimage import exposure

def get_dataset_info (dataset_path):
    list_images = []
    list_labels = []
    for i in os.listdir(os.path.join(dataset_path, "pneumonia")):
        list_images.append(os.path.join(dataset_path, "pneumonia", i))
        list_labels.append(0)
    
    for i in os.listdir(os.path.join(dataset_path, "covid")):
        list_images.append(os.path.join(dataset_path, "covid", i))
        list_labels.append(1)

    for i in os.listdir(os.path.join(dataset_path, "normal")):
        list_images.append(os.path.join(dataset_path, "normal", i))
        list_labels.append(2)


    list_images, list_labels = shuffle(list_images, list_labels)

    return list_images, list_labels

class COVID19_Dataset(Dataset):

    def __init__ (self, list_images, list_labels,  transform=None):

        self.list_images = list_images
        self.list_labels = list_labels
        self.len = len(self.list_images)
        self.mean = None
        self.std = None

        # Create dict of classes
        self.classes = ['pneumonia', 'covid', 'normal']
        self.num_classes = len(self.classes)
       
        
        self.weight_class = 1. / np.unique(np.array(self.list_labels), return_counts=True)[1]
        self.samples_weights = self.weight_class[self.list_labels]
        self.transform = transform
        #self.aug = aug

    def __len__(self):
        return self.len

    def weight(self):
        return self.weight_class

    def __getitem__(self, index):
        img_path = self.list_images[index]
        lbl = self.list_labels[index]

        img = imread(img_path, 1)
        img = ((img - np.min(img)) / (np.max(img) - np.min(img)))
#         img = exposure.equalize_adapthist(img)

        if img.shape[0] != 224 or img.shape[1] != 224:
            img = scikit_transform.resize(img, (224,224)).astype(img.dtype)

        
        img = img[:, :, None]

        # Apply transform 
        if self.transform:
            img = self.transform(img).float()
            #img = self.transform(T.functional.to_pil_image(img)).float()
        
        
        return img, lbl




