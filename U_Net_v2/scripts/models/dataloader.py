import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class Segmentation_Dataset(Dataset):
   
    def __init__(self, images_path: List[str], masks_path: List[str], width: int = 256, height: int = 256,
                 is_skeleton: bool = True, is_ridge_white: bool = True, dilate_kernel_size: int = 10):
        
        self.images_path = images_path
        self.masks_path = masks_path
        self.width = width
        self.height = height
        self.n_samples = len(images_path)
        self.is_skeleton = is_skeleton
        self.is_ridge_white = is_ridge_white
        self.dilate_kernel_size = dilate_kernel_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Reading and processing the image
        image = cv2.imread(self.images_path[index].as_posix(), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.width, self.height))  
        image = image / 255.0 
        image = np.transpose(image, (2, 0, 1))  # Transpose image to (C, H, W)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        # Reading and processing the mask
        mask = cv2.imread(self.masks_path[index].as_posix(), cv2.IMREAD_GRAYSCALE) 
        mask = cv2.bitwise_not(mask)
        kernel = np.ones((5, 5), np.uint8)  # You can adjust the size
        mask = cv2.erode(mask, kernel, iterations=1)

        # Create mask from skeleton labels
        if self.is_skeleton:

            if not self.is_ridge_white:
                mask = cv2.bitwise_not(mask)

            kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
            thickened_image = cv2.dilate(mask, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(thickened_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            biggest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

            mask = np.zeros_like(mask)
            cv2.drawContours(mask, [biggest_contour], -1, 255, thickness=cv2.FILLED)
        
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension to mask: (1, H, W)
        mask = mask / 255.0
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self) -> int:
        
        return self.n_samples
