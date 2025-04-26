import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import random

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(256, 256), 
                 normalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 augmentation=False, augmentation_prob=0.5):
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory '{image_dir}' does not exist!")
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory '{mask_dir}' does not exist!")

        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Ensure both image and mask files are sorted in the same order
        self.image_paths.sort()
        self.mask_paths.sort()

        self.image_size = image_size
        self.augmentation = augmentation
        self.augmentation_prob = augmentation_prob
        
        # Image transformations
        image_transforms = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
        
        # Add normalization for images if requested
        if normalize:
            image_transforms.append(transforms.Normalize(mean=mean, std=std))
            
        self.image_transform = transforms.Compose(image_transforms)
        
        # Mask transformations (no normalization for masks)
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # Assuming the mask is single channel
        
        # Apply augmentation if enabled
        if self.augmentation and random.random() < self.augmentation_prob:
            image, mask = self._apply_augmentation(image, mask)
        
        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        return image, mask
    
    def _apply_augmentation(self, image, mask):
        # Random horizontal flipping
        if random.random() < 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # Random vertical flipping
        if random.random() < 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
            
        # Random rotation
        if random.random() < 0.5:
            angle = random.randint(-30, 30)
            image = transforms.functional.rotate(image, angle)
            mask = transforms.functional.rotate(mask, angle)
            
        return image, mask
