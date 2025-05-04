import os
import cv2
import numpy as np
from empatches import EMPatches
from collections import defaultdict
import re

# Initialize EMPatches
emp = EMPatches()

# Directory containing patch images
patch_dir = '/home/ubuntu/m15kh/Image_Segmentation/inference_patch/masks'

# Directory to save merged images
output_dir = '/home/ubuntu/m15kh/Image_Segmentation/inference_patch/merged_masks'
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Dictionary to hold patches grouped by base image identifier
patch_groups = defaultdict(list)

# Iterate over files in the directory
for filename in os.listdir(patch_dir):
    if filename.endswith('.png'):
        # Extract base identifier (e.g., '30331320240684_porg_1')
        base_id = '_'.join(filename.split('_')[:3])
        patch_groups[base_id].append(filename)

# Process each group of patches
for base_id, files in patch_groups.items():
    # Sort files to maintain consistent order
    files.sort()

    # Load images
    patches = []
    for file in files:
        img = cv2.imread(os.path.join(patch_dir, file), cv2.IMREAD_GRAYSCALE)
        patches.append(img)

    # Determine patch size from the first patch
    patch_size = patches[0].shape[0]  # Assuming square patches

    # Count number of patches to determine grid size
    num_patches = len(patches)
    grid_size = int(np.ceil(np.sqrt(num_patches)))  # Determine grid dimensions
    
    # Calculate dimensions for the final merged image
    original_height = grid_size * patch_size
    original_width = grid_size * patch_size
    
    print(f"Base ID: {base_id}, Number of patches: {num_patches}, Grid size: {grid_size}x{grid_size}")
    print(f"Creating merged image with dimensions: {original_height}x{original_width}")
    
    # Create indices for the patches based on a grid layout
    indices = []
    for i, file in enumerate(files):
        # Extract patch index from filename using regex
        # This will work with filenames like "30331320240684_porg_0_patch_0.png"
        match = re.search(r'_patch_(\d+)\.png$', file)
        if match:
            patch_idx = int(match.group(1))
        else:
            patch_idx = i  # Fall back to the iteration index
        
        # Calculate row and column in the grid
        row = patch_idx // grid_size
        col = patch_idx % grid_size
        
        # Calculate top-left coordinates
        y_start = row * patch_size
        x_start = col * patch_size
        
        print(f"Placing patch {i} (from file {file}) at position ({y_start}, {x_start})")
        
        # Add the index tuple for this patch
        indices.append((y_start, x_start))

    # Use EMPatches to merge the patches
    try:
        merged_image = emp.merge_patches(patches, indices, mode='overwrite')
        
        # Convert back to uint8 for image saving
        merged_image = merged_image.astype(np.uint8)

        # Save the merged image
        output_path = os.path.join(output_dir, f'{base_id}_merged.png')
        cv2.imwrite(output_path, merged_image)
        print(f'Merged image saved at: {output_path}')
    except Exception as e:
        print(f"Error merging patches for {base_id}: {e}")
        
        # Fallback method if the library method fails
        print("Trying fallback merge method...")
        merged_image = np.zeros((original_height, original_width), dtype=np.uint8)
        for patch, (y_start, x_start) in zip(patches, indices):
            try:
                merged_image[y_start:y_start+patch_size, x_start:x_start+patch_size] = patch
            except Exception as e2:
                print(f"Error placing patch at ({y_start}, {x_start}): {e2}")
        
        output_path = os.path.join(output_dir, f'{base_id}_merged.png')
        cv2.imwrite(output_path, merged_image)
        print(f'Merged image saved using fallback method at: {output_path}')
