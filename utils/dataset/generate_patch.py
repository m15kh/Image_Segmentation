import os
import cv2
import numpy as np
from empatches import EMPatches
from tqdm import tqdm

# Define input directories
images_dir = '/home/rteam2/m15kh/FPR_dataset/Finger_Data/inference_test/images'
masks_dir = '/home/rteam2/m15kh/FPR_dataset/Finger_Data/inference_test/masks'

# Define output directories
output_base_dir = '/home/rteam2/m15kh/FPR_dataset/inference_patch'
images_patches_dir = os.path.join(output_base_dir, 'images')
masks_patches_dir = os.path.join(output_base_dir, 'masks')

# Create output directories if they don't exist
os.makedirs(images_patches_dir, exist_ok=True)
os.makedirs(masks_patches_dir, exist_ok=True)

# Initialize the EMPatches object
emp = EMPatches()

# Get all files from images directory
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))])
mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))])

# Process each image-mask pair
for img_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
    # Read image and mask
    img = cv2.imread(os.path.join(images_dir, img_file))
    mask = cv2.imread(os.path.join(masks_dir, mask_file))
    
    if img is None or mask is None:
        print(f"Error reading {img_file} or {mask_file}, skipping...")
        continue
    
    # Extract base filename without extension
    base_name = os.path.splitext(img_file)[0]
    

    # Extract patches
    img_patches, indices = emp.extract_patches(img, patchsize=1600, overlap=0)
    mask_patches, _ = emp.extract_patches(mask, patchsize=1600, overlap=0)
    
    # Save each patch
    for i, (img_patch, mask_patch, idx) in enumerate(zip(img_patches, mask_patches, indices)):
        # Create patch filenames
        img_patch_filename = f"{base_name}_patch_{i}.png"
        mask_patch_filename = f"{base_name}_patch_{i}.png"
        
        # Save patches
        cv2.imwrite(os.path.join(images_patches_dir, img_patch_filename), img_patch)
        cv2.imwrite(os.path.join(masks_patches_dir, mask_patch_filename), mask_patch)
    
print(f"Processing complete. Patches saved to {images_patches_dir} and {masks_patches_dir}")
