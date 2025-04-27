import os
import cv2
import math
import logging
import random
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from belong_mobile.models.DiffGAN.utils.loss import convert_to_one_channel

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def set_seed(seed: int):
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    # np.random.seed(seed) 
    # random.seed(seed)  

def random_z_generator(x, seed):
    torch.manual_seed(seed)  
    return torch.randn((1, x.shape[-2], x.shape[-1]), device=x.device).repeat(x.shape[0],1,1,1)


def random_latent_z_generator(x, seed, nz):
    torch.manual_seed(seed)  
    return torch.randn((1, nz), device=x.device).repeat(x.shape[0],1)

def sample_posterior(coefficients, x_0,x_t, t, device):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1.to(device), t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2.to(device), t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance.to(device), t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped.to(device), t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        # utils.set_seed(self.config.seed)
        noise = torch.randn([1, x_t.shape[-2], x_t.shape[-1]], device=x_t.device).repeat(x_t.shape[0],1,1,1)
        # noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos


def sample_posterior_seed(coefficients, x_0,x_t, t, seed, device):
    
    def q_posterior(x_0, x_t, t):
        mean = (
            extract(coefficients.posterior_mean_coef1.to(device), t, x_t.shape) * x_0
            + extract(coefficients.posterior_mean_coef2.to(device), t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance.to(device), t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped.to(device), t, x_t.shape)
        return mean, var, log_var_clipped
    
  
    def p_sample(x_0, x_t, t, seed):
        mean, _, log_var = q_posterior(x_0, x_t, t)
        # utils.set_seed(self.config.seed)
        noise = random_z_generator(x_t, seed)
        # noise = torch.randn([1, x_t.shape[-2], x_t.shape[-1]], device=x_t.device).repeat(x_t.shape[0],1,1,1)
        # noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t, seed)
    
    return sample_x_pos


def pad_image(image, patch_size):
    width, height = image.size
    pad_width = math.ceil(width / patch_size) * patch_size
    pad_height = math.ceil(height / patch_size) * patch_size
    padded_image = Image.new("RGB", (pad_width, pad_height), color="white")
    padded_image.paste(image, (0, 0))
    return padded_image

def pad_to_min_size(img, size, padding_color):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    old_width, old_height = img.size
    new_width = max(old_width, size)
    new_height = max(old_height, size)

    new_img = Image.new("L", (new_width, new_height), padding_color)
    pad_left = (new_width - old_width) // 2
    pad_top = (new_height - old_height) // 2

    new_img.paste(img, (pad_left, pad_top))
    return pad_left, pad_top, new_img

def pad_to_min_size_cv2(img, size, padding_color):
    # Ensure the input is a NumPy array
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Get old dimensions
    old_height, old_width = img.shape[:2]
    new_width = max(old_width, size)
    new_height = max(old_height, size)

    # Create a new image filled with the padding color
    new_img = np.full((new_height, new_width), padding_color, dtype=np.uint8)  # Grayscale image

    # Calculate padding
    pad_left = (new_width - old_width) // 2
    pad_top = (new_height - old_height) // 2

    # Paste the original image into the new image
    new_img[pad_top:pad_top + old_height, pad_left:pad_left + old_width] = img

    return pad_left, pad_top, new_img

def remove_padding(img, pad_left, pad_top, size):
    """
    Remove padding from an image based on the given padding values and original size.
    
    Parameters:
    - img: A PIL.Image object representing the padded image.
    - pad_left: The left padding amount added to the image.
    - pad_top: The top padding amount added to the image.
    - size: A tuple (original_width, original_height) representing the original size of the image before padding.
    
    Returns:
    - cropped_img: A PIL.Image object representing the image after removing padding.
    """
    original_width, original_height = size
    # Get the dimensions of the padded image
    padded_width, padded_height = img.size
    
    # Calculate the cropping box
    left = pad_left
    top = pad_top
    right = left + original_width
    bottom = top + original_height
    
    # Ensure the cropping box is within the image bounds
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, padded_width)
    bottom = min(bottom, padded_height)
    
    # Ensure the cropping box is valid
    if left >= right or top >= bottom:
        raise ValueError("Invalid cropping box. Ensure padding values and original size are correct.")
    
    # Crop the image to remove padding    
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def remove_padding_cv2(img, pad_left, pad_top, size):
    """
    Remove padding from an image based on the given padding values and original size.
    
    Parameters:
    - img: A NumPy array representing the padded image.
    - pad_left: The left padding amount added to the image.
    - pad_top: The top padding amount added to the image.
    - size: A tuple (original_width, original_height) representing the original size of the image before padding.
    
    Returns:
    - cropped_img: A NumPy array representing the image after removing padding.
    """
    original_width, original_height = size
    # Get the dimensions of the padded image
    padded_height, padded_width = img.shape[:2]
    
    # Calculate the cropping box
    left = pad_left
    top = pad_top
    right = left + original_width
    bottom = top + original_height
    
    # Ensure the cropping box is within the image bounds
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, padded_width)
    bottom = min(bottom, padded_height)
    
    # Ensure the cropping box is valid
    if left >= right or top >= bottom:
        raise ValueError("Invalid cropping box. Ensure padding values and original size are correct.")
    
    # Crop the image to remove padding    
    cropped_img = img[top:bottom, left:right]  # OpenCV uses (y, x) indexing
    return cropped_img


def pad_image_palm(image, patch_size):
    width, height = image.size
    pad_width = math.ceil(width / patch_size) * patch_size
    pad_height = math.ceil(height / patch_size) * patch_size
    padded_image = Image.new("RGB", (pad_width, pad_height), color="white")
    padded_image.paste(image, (0, 0))
    return padded_image

def convert_to_patch_palm(image, train_transform, patch_size=256, overlap=64):
    image = pad_image(image, patch_size)
    # Get the dimensions of the padded image
    width, height = image.size
    
    # Calculate the number of patches in both dimensions
    step = patch_size - overlap
    num_patches_x = (width - overlap) // step
    num_patches_y = (height - overlap) // step
    
    patches = []
    # Iterate over each patch and extract it
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Calculate the coordinates of the patch
            left = i * step
            upper = j * step
            right = left + patch_size
            lower = upper + patch_size
            
            # Extract the patch
            patch_tensor = image.crop((left, upper, right, lower))          

            patch_tensor = patch_tensor.convert("L")
            patch_tensor = train_transform(patch_tensor)
            patches.append(patch_tensor)

    return patches, num_patches_x, num_patches_y, step


def convert_to_patch(image, train_transform, patch_size=384):
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

    patches = []

    for i in range(num_patches_x):
        left = i * step_x
        if left + patch_size > width:
            left = width - patch_size
        for j in range(num_patches_y):
            upper = j * step_y
            if upper + patch_size > height:
                upper = height - patch_size

            # Extract the patch
            patch_tensor = image.crop((left, upper, left + patch_size, upper + patch_size))
            patch_tensor = patch_tensor.convert("L")
            patch_tensor = train_transform(patch_tensor)
            patches.append(patch_tensor)

    return patches, num_patches_x, num_patches_y, step_x, step_y



def reconstruct_image_palm(patches, patch_size, num_patches_x, num_patches_y, step):
    # Initialize an empty image and count array
    width = (num_patches_x - 1) * step + patch_size
    height = (num_patches_y - 1) * step + patch_size
    
    reconstructed_image = np.zeros((height, width, 3), dtype=np.float32)
    count = np.zeros((height, width, 3), dtype=np.float32)
    
    # Iterate over the patches and reconstruct the image
    patch_index = 0
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Calculate the coordinates of the patch
            left = i * step
            upper = j * step
            
            # Convert the patch tensor to a numpy array
            patch_tensor = torch.tensor(patches[patch_index])
            patch_image = patch_tensor.numpy().transpose(1, 2, 0)
            
            # Add the patch to the reconstructed image and increment the count
            reconstructed_image[upper:upper + patch_size, left:left + patch_size] += patch_image
            count[upper:upper + patch_size, left:left + patch_size] += 1
            
            # Increment the patch index
            patch_index += 1
    
    # Avoid division by zero
    count[count == 0] = 1
    
    # Average the overlapping regions
    reconstructed_image /= count
    
    # Convert the result to a PIL image
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
    reconstructed_image = Image.fromarray(reconstructed_image)
    
    return reconstructed_image


def reconstruct_image(patches, patch_size, num_patches_x, num_patches_y, step_x, step_y, original_size):
    original_width, original_height = original_size

    # Initialize arrays for reconstruction and count
    reconstructed_image = np.zeros((original_height, original_width), dtype=np.float32)
    count = np.zeros((original_height, original_width), dtype=np.float32)

    # Iterate over the patches and reconstruct the image
    patch_index = 0
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Calculate the coordinates of the patch
            left = i * step_x
            upper = j * step_y

            # Handle the edge cases
            if left + patch_size > original_width:
                left = original_width - patch_size
            if upper + patch_size > original_height:
                upper = original_height - patch_size

            right = left + patch_size
            lower = upper + patch_size

            # Extract the patch from the tensor
            patch_tensor = patches[patch_index]
            patch_image = patch_tensor.cpu().numpy().squeeze()  # Remove channel dimension if present

            # Add the patch to the reconstructed image and increment the count
            reconstructed_image[upper:lower, left:right] += patch_image[:lower-upper, :right-left]
            count[upper:lower, left:right] += 1

            # Increment the patch index
            patch_index += 1

    # Avoid division by zero
    count[count == 0] = 1

    # Average the overlapping regions
    reconstructed_image /= count

    # Convert the result to a PIL image
    reconstructed_image = np.clip(reconstructed_image, 0, 1)  # Ensure values are within [0, 1]
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  # Scale to [0, 255]
    reconstructed_image = Image.fromarray(reconstructed_image, mode='L')

    return reconstructed_image

