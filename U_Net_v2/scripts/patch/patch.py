import os
from empatches import EMPatches, patch_via_indices
import cv2
import numpy as np
from tqdm import tqdm  # nice progress bar (pip install tqdm if missing)

# Set your folders
data_folder = '/home/fteam6/m15kh/U_NET/U_Net_v2/data/data_test/test'      # folder containing input images
mask_folder = '/home/fteam6/m15kh/U_NET/U_Net_v2/data/data_test/mask_data'      # folder containing mask images
new_data_folder = '/home/fteam6/m15kh/U_NET/U_Net_v2/new_data'
new_mask_folder = '/home/fteam6/m15kh/U_NET/U_Net_v2/new_maks'

# Create new folders if they don't exist
os.makedirs(new_data_folder, exist_ok=True)
os.makedirs(new_mask_folder, exist_ok=True)

# Initialize EMPatches
emp = EMPatches()

# List all data images
data_images = sorted(os.listdir(data_folder))

# Loop through each image
for img_name in tqdm(data_images):
    # Full paths
    data_path = os.path.join(data_folder, img_name)
    mask_path = os.path.join(mask_folder, img_name)  # Assuming mask has the same filename!

    # Check if the mask file exists
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Error: Corresponding mask file '{img_name}' does not exist in '{mask_folder}'.")

    # Load images
    data_img = cv2.imread(data_path)
    data_img = cv2.cvtColor(data_img, cv2.COLOR_BGR2RGB)

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # or remove IMREAD_GRAYSCALE if RGB mask

    # Extract patches
    data_patches, indices = emp.extract_patches(data_img, patchsize=1500, overlap=0.1)
    mask_patches = patch_via_indices(mask_img, indices)

    # Save each patch
    for i, (data_patch, mask_patch) in enumerate(zip(data_patches, mask_patches)):
        # Save data patch
        data_patch = cv2.cvtColor(data_patch, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
        data_patch_name = f"{os.path.splitext(img_name)[0]}_patch_{i:03d}.png"
        cv2.imwrite(os.path.join(new_data_folder, data_patch_name), data_patch)

        # Save mask patch
        mask_patch_name = f"{os.path.splitext(img_name)[0]}_patch_{i:03d}.png"
        cv2.imwrite(os.path.join(new_mask_folder, mask_patch_name), mask_patch)

print("All patches saved successfully!")
