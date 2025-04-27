import torch.utils.data as data
import numpy as np
# import lmdb
import os
import io
import pickle
from PIL import Image
import glob
import cv2
from pathlib import Path
import torch


class DataLoaderTrain(data.Dataset):

    def __init__(self, root, image_extentions, image_size, transform = None, transform_mask = None, crop_image = False, use_minutiae = True):
        images = []
        for image_p in glob.glob(root + f"/*{image_extentions}"):
            if "perspective" not in image_p:
                images.append(image_p)
        self.train_images = images[:int(0.99 * len(images))]

        self.transform = transform
        self.transform_mask = transform_mask
        self.image_size = image_size
        self.crop_image = crop_image
        self.use_minutiae = use_minutiae

    def skeleton_to_mask(self, image: np.ndarray, is_ridge_white: bool):
        if is_ridge_white:
            image = cv2.bitwise_not(image)
        _, thresholded_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thickened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thickened_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        binary_mask = np.ones_like(image)
        cv2.drawContours(binary_mask, contours, -1, (0), thickness=cv2.FILLED)
        return binary_mask

    def mask_to_boundingbox(self, mask: np.array):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(sorted_contours[0])
        return x, y, w, h         

    def __getitem__(self, index):

        image_path = self.train_images[index]
        template_path = image_path.replace("images", "masks")  
        # if self.use_minutiae:      
        #     mpoint_mask = image_path.replace("images", "mpoints") 
        # orientation_mask = image_path.replace("images", "orientations") #orientation


        img1 = cv2.imread(image_path, 0)
        img1 = Image.fromarray(img1)


        img2 = cv2.imread(template_path, 0)
        img2 = cv2.resize(img2, (self.image_size, self.image_size))     
        img2 = (img2 > 0) * 255
        img2 = np.array(img2, dtype=np.uint8)
        img2 = Image.fromarray(img2)
        
        # if self.use_minutiae:
        #     img3 = cv2.imread(mpoint_mask, 0)
        #     img3 = cv2.resize(img3, (self.image_size, self.image_size))     
        #     img3 = (img3 > 0) * 255
        #     img3 = np.array(img3, dtype=np.uint8)
        #     img3 = Image.fromarray(img3)       
        
        # img4 = cv2.imread(orientation_mask, 0)   
        # img4 = cv2.resize(img4, (self.image_size, self.image_size))  
        # img4 = (img4 > 0) * 255
        # img4 = np.array(img4, dtype=np.uint8)        
        # img4 = Image.fromarray(img4)         

        if self.crop_image:
            x, y, w, h = self.mask_to_boundingbox(self.skeleton_to_mask(cv2.imread(image_path, 0), is_ridge_white = False) * 255)
            img1 = img1.crop((x, y, x + w, y + h))  
            img2 = img2.crop((x, y, x + w, y + h))  
            # if self.use_minutiae:
            #     img3 = img3.crop((x, y, x + w, y + h))  
            # img4 = img4.crop((x, y, x + w, y + h))  

            
        if self.transform is not None:
            img1 = self.transform(img1)
        if self.transform_mask is not None:            
            img2 = self.transform_mask(img2)  
            # if self.use_minutiae:
            #     img3 = self.transform_mask(img3)  
            # img4 = self.transform_mask(img4)     

        # if self.use_minutiae:
        #     return img1, img2#, img3, img4 
        # else:
        #     return img1, img2#, img4
        return img1, img2#, img4 #@Borhan


    def __len__(self):
        return len(self.train_images)


class DataLoaderTest(data.Dataset):
    def __init__(self, root, image_extentions, image_size, transform = None, transform_mask = None, crop_image = False, use_minutiae = True):
        images = []
        for image_p in glob.glob(root + f"/*{image_extentions}"):
            if "perspective" not in image_p:
                images.append(image_p)        
        self.train_images = images[int(0.99 * len(images)):]
        self.transform = transform
        self.transform_mask = transform_mask
        self.image_size = image_size
        self.use_minutiae = use_minutiae
        self.crop_image = crop_image

    def skeleton_to_mask(self, image: np.ndarray, is_ridge_white: bool):
        if is_ridge_white:
            image = cv2.bitwise_not(image)
        _, thresholded_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        thickened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thickened_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        binary_mask = np.ones_like(image)
        cv2.drawContours(binary_mask, contours, -1, (0), thickness=cv2.FILLED)
        return binary_mask

    def mask_to_boundingbox(self, mask: np.array):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(sorted_contours[0])
        return x, y, w, h        

    def __getitem__(self, index):

        image_path = self.train_images[index]
        template_path = image_path.replace("images", "masks")
        if self.use_minutiae:
            mpoint_mask = image_path.replace("images", "mpoints")
        orientation_mask = image_path.replace("images", "orientations")#orientation

        img1 = cv2.imread(image_path, 0)
        img1 = Image.fromarray(img1)

        img2 = cv2.imread(template_path, 0)   
        img2 = cv2.resize(img2, (self.image_size, self.image_size))  
        img2 = (img2 > 0) * 255
        img2 = np.array(img2, dtype=np.uint8)        
        img2 = Image.fromarray(img2)

        if self.use_minutiae:
            img3 = cv2.imread(mpoint_mask, 0)   
            img3 = cv2.resize(img3, (self.image_size, self.image_size))  
            img3 = (img3 > 0) * 255
            img3 = np.array(img3, dtype=np.uint8)        
            img3 = Image.fromarray(img3)


        img4 = cv2.imread(orientation_mask, 0)   
        img4 = cv2.resize(img4, (self.image_size, self.image_size))  
        img4 = (img4 > 0) * 255
        img4 = np.array(img4, dtype=np.uint8)        
        img4 = Image.fromarray(img4)


        if self.crop_image:
            x, y, w, h = self.mask_to_boundingbox(self.skeleton_to_mask(cv2.imread(image_path, 0), is_ridge_white = False) * 255)
            img1 = img1.crop((x, y, x + w, y + h))  
            img2 = img2.crop((x, y, x + w, y + h))  
            if self.use_minutiae:
                img3 = img3.crop((x, y, x + w, y + h))  
            img4 = img4.crop((x, y, x + w, y + h))  
        
        

        if self.transform is not None:
            img1 = self.transform(img1)
        if self.transform_mask is not None:            
            img2 = self.transform_mask(img2) 
            if self.use_minutiae:  
                img3 = self.transform_mask(img3)  
            img4 = self.transform_mask(img4)  
        if self.use_minutiae:      
            return img1, img2, img3, img4
        else:
            return img1, img2, img4

    def __len__(self):
        return len(self.train_images)        