import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os

class LFWDataset(Dataset):
    def __init__(
        self,
        faces_folder,
        smiling_labels_file,
        non_smiling_labels_file,
        transform=None
    ):
        self.faces_folder = faces_folder
        self.transform = transform

        # Read the smiling faces file and extract filenames
        with open(smiling_labels_file, 'r') as f:
            smiling_files = [
                line.strip().replace('.jpg', '.ppm') 
                for line in f.readlines() 
                if line.strip() and not line.strip().endswith('listt.txt')
            ]

        # Read the non-smiling faces file and extract filenames
        with open(non_smiling_labels_file, 'r') as f:
            non_smiling_files = [
                line.strip().replace('.jpg', '.ppm') 
                for line in f.readlines() 
                if line.strip()
            ]

        # Create image paths and labels
        self.image_paths = []
        self.labels = []

        # Add smiling faces to the dataset with label = 1
        for filename in smiling_files:
            image_path = os.path.join(self.faces_folder, filename)
            if os.path.isfile(image_path):
                self.image_paths.append(image_path)
                self.labels.append(1)

        # Add non-smiling faces to the dataset with label = 0
        for filename in non_smiling_files:
            image_path = os.path.join(self.faces_folder, filename)
            if os.path.isfile(image_path):
                self.image_paths.append(image_path)
                self.labels.append(0)

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path and label for the given index
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load and convert the image to RGB
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label