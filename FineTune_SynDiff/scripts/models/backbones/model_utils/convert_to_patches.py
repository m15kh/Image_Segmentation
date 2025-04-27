import sys
import argparse
from pathlib import Path
ROOT_DIR = Path(__file__).parents[4].as_posix()
# print(ROOT_DIR)
sys.path.append(ROOT_DIR)
import os
import cv2
import glob
import math
import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader



from src.rembg.diff_gan.utils import utils

class CreateDataset:

    def __init__(self, images_path, orientation_extractor, use_mpoints, image_size,
                  mpoint_extention, mask_extention, image_extention,
                  bg_mask_color):
        self.images_path           = images_path
        self.orientation_extractor = orientation_extractor
        self.use_mpoints           = use_mpoints               
        self.size                  = image_size 
        self.mpoint_extention      = mpoint_extention.replace(".", "")
        self.mask_extention        = mask_extention.replace(".", "")
        self.image_extention       = image_extention.replace(".", "") 
        self.bg_mask_color         = bg_mask_color               
        os.makedirs("enahancer_training_data_temp/images", exist_ok=True)
        os.makedirs("enahancer_training_data_temp/masks", exist_ok=True)
        os.makedirs("enahancer_training_data_temp/orientations", exist_ok=True)

        if self.use_mpoints:
            os.makedirs("enahancer_training_data_temp/mpoints", exist_ok=True)

        



    def convert_to_patch(self, image, mask, mnt, orientations, patch_size=384):
        width, height = image.size
        # Calculate the number of patches in both dimensions
        if width <= patch_size:
            step_x = width
            num_patches_x = 1
        else:
            num_patches_x = (width - patch_size) // (patch_size // 2) + 1
            step_x = (width - patch_size) // (num_patches_x - 1)
        
        if height <= patch_size:
            step_y = height
            num_patches_y = 1
        else:
            num_patches_y = (height - patch_size) // (patch_size // 2) + 1
            step_y = (height - patch_size) // (num_patches_y - 1)

        if mnt != None:
            image_patches, mask_patches, mnt_patches, orientations_patches = [], [], [], []
        else:
            image_patches, mask_patches, orientations_patches = [], [], []      

        for i in range(num_patches_x):
            left = i * step_x
            if left + patch_size > width:
                left = width - patch_size
            for j in range(num_patches_y):
                upper = j * step_y
                if upper + patch_size > height:
                    upper = height - patch_size

                # Extract the image patch
                patch_tensor = image.crop((left, upper, left + patch_size, upper + patch_size))
                patch_tensor = patch_tensor.convert("L")
                image_patches.append(patch_tensor)


                # Extract the mask patch
                mask_tensor = mask.crop((left, upper, left + patch_size, upper + patch_size))
                mask_tensor = mask_tensor.convert("L")
                mask_patches.append(mask_tensor)            

                # Extract the mpoint-mask patch
                if mnt != None:
                    mnt_tensor = mnt.crop((left, upper, left + patch_size, upper + patch_size))
                    mnt_tensor = mnt_tensor.convert("L")
                    mnt_patches.append(mnt_tensor)          

                # Extract the orientation patch
                orientations_tensor = orientations.crop((left, upper, left + patch_size, upper + patch_size))
                orientations_tensor = orientations_tensor.convert("L")
                orientations_patches.append(orientations_tensor)                          

        if mnt != None:
            return image_patches, mask_patches, mnt_patches, orientations_patches
        else:
            return image_patches, mask_patches, orientations_patches

    def create_mnt(self, path, mask_shape, radius=12, color=255, thickness=-1):
        if self.mpoint_extention == "npy":
            mnts = np.load(path)
            black_mask = np.zeros(mask_shape, dtype=np.uint8)
            x_coords, y_coords = mnts[:, 0].astype(int), mnts[:, 1].astype(int)
            for x, y in zip(x_coords, y_coords):
                cv2.circle(black_mask, (x, y), radius, color, thickness)
        if self.mpoint_extention == "mnt":
            df = pd.read_csv(path)            
            black_mask = np.zeros(mask_shape, dtype=np.uint8)
            x_coords = df["x"].tolist()
            y_coords = df["y"].tolist()
            for x, y in zip(x_coords, y_coords):
                cv2.circle(black_mask, (int(x), int(y)), radius, color, thickness)            
        return black_mask

    def pad_to_min_size(self, img, size):

        old_width, old_height = img.size
        
        new_width = max(old_width, size)
        new_height = max(old_height, size)
        
        new_img = Image.new("L", (new_width, new_height), (255 if self.bg_mask_color == 0 else 0))
        
        # Calculate the padding values to center the original image
        pad_left = (new_width - old_width) // 2
        pad_top = (new_height - old_height) // 2
        
        # Paste the original image onto the center of the new image
        new_img.paste(img, (pad_left, pad_top))
        
        
        # Return the padding sizes
        return pad_left, pad_top, new_img

    def enhanced_image_cleaner(self, image,  mask_type="poly"):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        blur = cv2.GaussianBlur(image, (5, 5), sigmaX=0, sigmaY=0)
        gX = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
        gY = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
        magnitude = np.sqrt((gX ** 2) + (gY ** 2)).astype("uint8")
        mask, big_contour = self.get_elips_dynamic(magnitude.astype("uint8"), mask_type)
        return mask  

    def get_elips_dynamic(self, gray, mask_type="poly"):
        gray = cv2.dilate(gray.copy(), None, iterations=5)
        mask = np.zeros_like(gray)
        thresh = cv2.threshold(gray, 100 , 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv2.contourArea)
        if mask_type == "poly":
            mask = self.find_poly(big_contour, mask)
        return mask, big_contour            

    def find_poly(self, big_contour, mask):
        msk = mask.copy()
        hull = cv2.convexHull(big_contour)
        msk = cv2.fillPoly(msk, [hull], (255,255,255))
        return msk

    def save_patches(self, is_csv):
        i = 0
        for image_path in tqdm.tqdm(self.images_path,
                                               desc='Processing and Saving Images and Masks',
                                               unit='Image',
                                               ncols = 100,
                                               total = len(self.images_path)):            
            
            mask_path = os.path.join(Path(image_path).parents[1].as_posix(), "masks", Path(image_path).stem + f".{self.mask_extention}")                        
            orient_path = os.path.join(Path(image_path).parents[1].as_posix(), "orientations", Path(image_path).stem + f".{self.mask_extention}")                        
            assert os.path.isfile(mask_path), f"No mask file found in directory {mask_path}, please check mask file extention and folder directory in params/params_train.yaml"
                 
            if is_csv:
                image_path = image_path[1:]
                mask_path  = mask_path[1:]
            if "latent" not in image_path and ("_" + image_path.split(os.path.sep)[-1].split(".")[0].split("_")[-1] + ".png" not in ["_13.png", "_14.png"]):                
                image = Image.open(image_path)
                mask  = Image.open(mask_path) 
                 
                
                _, _, image = self.pad_to_min_size(image, self.size)
                _, _, mask  = self.pad_to_min_size(mask, self.size)

                step = 500 if image.size[0] * image.size[1] > 1e6 else 200
                

                #Be Advised: LatentDataset is just a name and converts any type of input fingerprint image into patches
                image_patcher = utils.LatentDataset(np.array(image), self.size, self.size, step)
                mask_patcher  = utils.LatentDataset(np.array(mask), self.size, self.size, step)
                    
                #Get Orientation Field
                if os.path.isfile(orient_path):
                    orientation  = Image.open(orient_path)
                    self.bg_mask_color = 0
                    _, _, orientation  = self.pad_to_min_size(orientation, self.size)
                    orientations_patcher  = utils.LatentDataset(np.array(orientation), self.size, self.size, step)
                else:
                    segmentation_mask = self.enhanced_image_cleaner(np.array(mask))
                    orientations = self.orientation_extractor.inference(np.array(mask), segmentation_mask)
                    orientations = self.orientation_extractor.draw_orientations(np.array(image), orientations, segmentation_mask
                                                                                , draw_on_white_image = True)
                    orientations = Image.fromarray(orientations.astype(np.uint8)).convert("L")
                    orientations_patcher  = utils.LatentDataset(np.array(orientations), self.size, self.size, step)
                #Read MNTs
                if self.use_mpoints:                                        
                    mnt_path = os.path.join(Path(mask_path).parents[1].as_posix(), "mpoints", Path(mask_path).stem + f".{self.mpoint_extention}")   
                    assert os.path.isfile(mnt_path), f"No mpoint file found in directory {mnt_path}, please check mpoint file extention and folder directory in params/params_train.yaml"                 
                    mnt_mask = self.create_mnt(mnt_path, mask.size)
                    mnt_mask = Image.fromarray(mnt_mask)
                    _, _, mnt_mask  = self.pad_to_min_size(mnt_mask, self.size)
                    mnt_patcher  = utils.LatentDataset(np.array(mnt_mask), self.size, self.size, step)
                    mnt_patches = DataLoader(mnt_patcher, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
                #Save patches
                image_patches = DataLoader(image_patcher, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
                mask_patches  = DataLoader(mask_patcher, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
                ori_patches   = DataLoader(orientations_patcher, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
        
                save_path_image = f"enahancer_training_data_temp/images/{i:06d}.png"
                save_path_mask  = f"enahancer_training_data_temp/masks/{i:06d}.png"
                save_path_oris  = f"enahancer_training_data_temp/orientations/{i:06d}.png"
                if self.use_mpoints:
                    save_path_mnts  = f"enahancer_training_data_temp/mpoints/{i:06d}.png"                
                
                if self.use_mpoints:
                    assert len(image_patches) == len(mask_patches) == len(mnt_patches) == len(ori_patches), "All extracted patches must have same length"
                else:
                    assert len(image_patches) == len(mask_patches) == len(ori_patches), "All extracted patches must have same length"

                #Ensure that blank masks do not save.
                blank_mask_index = []
                for step, (patch_ind, patch) in enumerate(mask_patches):                           
                    if len(torch.unique(patch)) != 1: 
                        blank_mask_index.append(0)
                    else:
                        blank_mask_index.append(1)                                
                #Save Image
                for step, (patch_ind, patch) in enumerate(image_patches):
                    if blank_mask_index[step] == 0:
                        patch = patch.cpu().detach().numpy().squeeze(0).squeeze(0)
                        patch = Image.fromarray(np.uint8(patch))
                        patch.save(save_path_image.replace(".png", "_patch_{}.png".format(step)))
                #Save Mask
                for step, (patch_ind, patch) in enumerate(mask_patches):
                    if blank_mask_index[step] == 0:
                        patch = patch.cpu().detach().numpy().squeeze(0).squeeze(0)
                        patch = Image.fromarray(np.uint8(patch))
                        patch.save(save_path_mask.replace(".png", "_patch_{}.png".format(step)))
                    
                #Save Ori
                for step, (patch_ind, patch) in enumerate(ori_patches):
                    if blank_mask_index[step] == 0:
                        patch = patch.cpu().detach().numpy().squeeze(0).squeeze(0)
                        patch = Image.fromarray(np.uint8(patch))
                        patch.save(save_path_oris.replace(".png", "_patch_{}.png".format(step)))
                    
                #Save MNTs
                if self.use_mpoints:
                    for step, (patch_ind, patch) in enumerate(mnt_patches):
                        if blank_mask_index[step] == 0:
                            patch = patch.cpu().detach().numpy().squeeze(0).squeeze(0)
                            patch = Image.fromarray(np.uint8(patch))
                            patch.save(save_path_mnts.replace(".png", "_patch_{}.png".format(step)))   
                        
                
                i += 1                                                                
                
          
    
    def save_full_image(self, is_csv):

        i = 0
        for image_path in tqdm.tqdm(self.images_path):
            # try:
            if is_csv:
                image_path = image_path[1:]
                mask_path  = mask_path[1:]
            if "latent" not in image_path and ("_" + image_path.split(os.path.sep)[-1].split(".")[0].split("_")[-1] + ".png" not in ["_13.png", "_14.png"]):
                # try:
                # if os.path.isfile(image_path):
                mask_path = image_path.replace("images", "masks").replace(f"{self.image_extention}", f"{self.mask_extention}")
                orient_path = os.path.join(Path(image_path).parents[1].as_posix(), "orientations", Path(image_path).stem + f".{self.mask_extention}")                        
                
                image = Image.open(image_path)
                mask  = Image.open(mask_path) 
                    
                #Get Orientation Field
                if os.path.isfile(orient_path):
                    orientations  = Image.open(orient_path)
                    # orientations = Image.fromarray(orientations.astype(np.uint8))
                else:
                    segmentation_mask = self.enhanced_image_cleaner(np.array(mask))
                    orientations = self.orientation_extractor.inference(np.array(image.convert('L')), segmentation_mask)
                    orientations = self.orientation_extractor.draw_orientations(np.array(image), orientations, segmentation_mask
                                                                                , draw_on_white_image = True)
                    orientations = Image.fromarray(orientations.astype(np.uint8))
                #Read MNTs
                if self.use_mpoints:
                    if is_csv:                        
                        mnt_path = mask_path.replace(".png", "_minutiapoints.npy")
                        mnt_mask = self.create_mnt(mnt_path, mask.size)
                        mnt_mask = Image.fromarray(mnt_mask)
                    else:
                        mnt_path = mask_path.replace("masks", "mpoints").replace(".png", f".{self.mpoint_extention}")
                        mnt_mask = self.create_mnt(mnt_path, mask.size)
                        mnt_mask = Image.fromarray(mnt_mask)

                #Save
                save_path_image = f"enahancer_training_data_temp/images/{i:06d}.png"
                save_path_mask  = f"enahancer_training_data_temp/masks/{i:06d}.png"
                if self.use_mpoints:
                    save_path_mnts  = f"enahancer_training_data_temp/mpoints/{i:06d}.png"
                save_path_oris  = f"enahancer_training_data_temp/orientations/{i:06d}.png"         

                image.save(save_path_image)
                mask.save(save_path_mask)
                if self.use_mpoints:
                    mnt_mask.save(save_path_mnts)
                orientations.save(save_path_oris)   
                i += 1
