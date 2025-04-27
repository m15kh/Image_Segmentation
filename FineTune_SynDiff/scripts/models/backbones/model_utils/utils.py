import os
import cv2
import math
import logging

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from belong_mobile.models.DiffGAN.backbones.model_utils.loss import convert_to_one_channel

def psnr(img1, img2):
    #Peak Signal to Noise Ratio

    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(img1.max() / torch.sqrt(mse))

        
#%% Diffusion coefficients 
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var

def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_time_schedule(args):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small)  + eps_small
    return t

def get_sigma_schedule(args):
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3
   
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    
    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    
    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas))
    betas = betas.type(torch.float32)
    sigmas = betas**0.5
    a_s = torch.sqrt(1-betas)
    return sigmas, a_s, betas

#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args):
        
        _, _, self.betas = get_sigma_schedule(args)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
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
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:,None,None,None] * torch.exp(0.5 * log_var) * noise
            
    sample_x_pos = p_sample(x_0, x_t, t)
    
    return sample_x_pos




def sample_from_model(coefficients, generator, n_time, x_init, T, opt):
    x = x_init[:,[0],:] #Noise
    source = x_init[:,[1],:] #real_image

    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64)
            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz)
            x_0 = generator(torch.cat((x, source),axis=1), t_time, latent_z)
            x_new = sample_posterior(coefficients, x_0[:,[0],:], x, t)
            x = x_new.detach()
        
    return x

def pad_image(image, patch_size):
    width, height = image.size
    pad_width = math.ceil(width / patch_size) * patch_size
    pad_height = math.ceil(height / patch_size) * patch_size
    padded_image = Image.new("RGB", (pad_width, pad_height), color="white")
    padded_image.paste(image, (0, 0))
    return padded_image

def pad_to_min_size(img, size):

    old_width, old_height = img.size
    
    new_width = max(old_width, size)
    new_height = max(old_height, size)
    
    new_img = Image.new("L", (new_width, new_height), (255))
    
    # Calculate the padding values to center the original image
    pad_left = (new_width - old_width) // 2
    pad_top = (new_height - old_height) // 2
    
    # Paste the original image onto the center of the new image
    new_img.paste(img, (pad_left, pad_top))
    
    
    # Return the padding sizes
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


class LatentDataset(Dataset):
    def __init__(self, latent, input_row, input_col, step):
        self.input_row = input_row
        self.input_col = input_col
        shape_latent = latent.shape        
        ROW = shape_latent[0]
        COL = shape_latent[1]
        row_list_1 = range(input_row, ROW+1, step)
        row_list_2 = range(ROW, row_list_1[-1]-1,-step)
        row_list = [*row_list_1, *row_list_2]
        
        col_list_1 = range(input_col, COL+1, step)
        col_list_2 = range(COL, col_list_1[-1]-1, -step)
        col_list = [*col_list_1,*col_list_2]
        
        self.num_patch = len(row_list)*len(col_list)

        row_col_inds = np.zeros([self.num_patch,2]).astype(np.int32)
        self.latent = latent.reshape(1, ROW,COL)
        ind = 0
        for row_ind in row_list:
            for col_ind in col_list:
                row_col_inds[ind,:] = [row_ind,col_ind]
                ind += 1

        self.row_col_inds = torch.from_numpy(row_col_inds)

    def __len__(self):
        return self.num_patch

    def __getitem__(self, ind):
        row_col = self.row_col_inds[ind].clone()
        row = row_col[0]
        col = row_col[1]
        patch = self.latent[:,(row-self.input_row):row,(col-self.input_col):col]
        return row_col, patch

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



def calculate_dice(mask1, mask2):

    # Ensure the masks are binary (0 or 1)
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Calculate intersection and the sum of the areas
    intersection = np.logical_and(mask1, mask2).sum()
    mask1_sum = mask1.sum()
    mask2_sum = mask2.sum()

    # Compute Dice coefficient
    dice = (2 * intersection) / (mask1_sum + mask2_sum)
    return dice


def calculate_iou(mask1, mask2):

    # Ensure the masks are binary (0 or 1)
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Compute IoU
    if union == 0:
        return 0.0  # Avoid division by zero

    iou = intersection / union
    return iou

