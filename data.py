from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
import pandas as pd
from torchvision import transforms
import torchvision

class train_dataset(Dataset):
    def __init__(self, csv_path='AID_data_train.csv'):
        csv = pd.read_csv(csv_path)
        self.image_paths = csv['image'].values
        self.labels = csv['label'].values

        self.transforms = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1)
        ]
        self.transforms_2=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # imagenet normalization
        ])
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        #get the image
        image = Image.open(self.image_paths[i])
        #get the label
        label = self.labels[i]
        # choose transformation
        transform = np.random.choice(self.transforms, size=1)[0]
        image_1 = self.transforms_2(image)
        # transform image
        image_2 = transform(self.transforms_2(image))
        # return original image, transformed image, label, transformation
        if type(transform) == torchvision.transforms.transforms.RandomHorizontalFlip:
            return image_1, image_2, label, 'horizontal'
        else:
            return image_1, image_2, label, 'vertical'
    


class val_dataset(Dataset):
    def __init__(self, csv_path='AID_data_val.csv'):
        csv = pd.read_csv(csv_path)
        self.image_paths = csv['image'].values
        self.labels = csv['label'].values

        self.transforms = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1)
        ]
        self.transforms_2=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # imagenet normalization
        ])
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        #get the image
        image = Image.open(self.image_paths[i])
        #get the label
        label = self.labels[i]
        # choose transformation
        transform = np.random.choice(self.transforms, size=1)[0]
        image_1 = self.transforms_2(image)
        # transform image
        image_2 = transform(self.transforms_2(image))
        # return original image, transformed image, label, transformation
        if type(transform) == torchvision.transforms.transforms.RandomHorizontalFlip:
            return image_1, image_2, label, 'horizontal'
        else:
            return image_1, image_2, label, 'vertical'


class triplet_dataset(Dataset):
    def __init__(self, csv_path='AID_data_triplet.csv'):
        csv = pd.read_csv(csv_path)
        self.anchor = csv['image'].values
        self.labels = csv['label'].values
        self.pos = csv['image_pos'].values
        self.neg = csv['image_neg'].values
        self.transforms = [
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1)
        ]
        self.transforms_2=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # imagenet normalization
        ])
        self.transforms_3 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # imagenet normalization
        ])
    
    def __len__(self):
        return len(self.anchor)
    
    def __getitem__(self, idx):
        #get the image
        image = Image.open(self.anchor[idx])
        #get the label
        label = self.labels[idx]
        # choose transformation
        transform = np.random.choice(self.transforms, size=1)[0]
        image_1 = self.transforms_2(image)
        # transform image
        image_2 = transform(self.transforms_2(image))
        # return original image, transformed image, label, transformation
        if type(transform) == torchvision.transforms.transforms.RandomHorizontalFlip:
            return image_1, image_2, label, 'horizontal',self.transforms_3(Image.open(self.pos[idx])), self.transforms_3(Image.open(self.neg[idx]))
        else:
            return image_1, image_2, label, 'vertical',self.transforms_3(Image.open(self.pos[idx])), self.transforms_3(Image.open(self.neg[idx]))
        return  

# class val_dataset(Dataset):
#     def __init__(self, csv_path='AID_val_data.csv'):
#         csv = pd.read_csv(csv_path)
#         self.image_paths = csv['image'].values
#         self.labels = csv['label'].values

#         self.transforms_2=transforms.Compose([
#             transforms.Resize((224,224)),
#             transforms.ToTensor(),
#             # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # imagenet normalization
#         ])
    
#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, i):
#         #get the image
#         image = Image.open(self.image_paths[i])
#         #get the label
#         label = self.labels[i]
#         # choose transformation
#         image = self.transforms_2(image)
#         # return original image, transformed image, label, transformation

#         return image, label