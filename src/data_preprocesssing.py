import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image


class DataPreprocessor:
    def __init__(self, data_dir, batch_size, image_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

    def create_data_loaders(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),  # Resize images to a consistent size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]
                                     )
            ]),
            'val': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),  # Resize images to a consistent size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),  # Resize images to a consistent size
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        image_datasets = {
            'train': ImageFolder(os.path.join(self.data_dir, 'train'), data_transforms['train']),
            'val': ImageFolder(os.path.join(self.data_dir, 'evaluation'), data_transforms['val']),
            'test': ImageFolder(os.path.join(self.data_dir, 'test'), data_transforms['test'])
        }

        data_loaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=True),
            'val': DataLoader(image_datasets['val'], batch_size=self.batch_size, shuffle=False),
            'test': DataLoader(image_datasets['test'], batch_size=self.batch_size, shuffle=False)
        }
        return data_loaders

