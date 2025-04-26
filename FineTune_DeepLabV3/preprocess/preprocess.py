import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to your dataset
image_dir = '/home/ubuntu/m15kh/U_NET/DeepLabV3/dataset/clean_images'  # Replace with the path to your images
mask_dir = '/home/ubuntu/m15kh/U_NET/DeepLabV3/dataset/masks'    # Replace with the path to your masks

# Output directories
output_dirs = {
    'train_images': '/home/ubuntu/m15kh/U_NET/DeepLabV3/fingerprint/train_images',
    'train_masks': '//home/ubuntu/m15kh/U_NET/DeepLabV3/fingerprint/train_masks',
    'valid_images': '/home/ubuntu/m15kh/U_NET/DeepLabV3/fingerprint/valid_images',
    'valid_masks': '/home/ubuntu/m15kh/U_NET/DeepLabV3/fingerprint/valid_masks'
}

# Create output directories if they don't exist
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Get list of image and mask files
image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

# Ensure images and masks are aligned
assert len(image_files) == len(mask_files), "Mismatch between images and masks count"
assert all(img.split('.')[0] == mask.split('.')[0] for img, mask in zip(image_files, mask_files)), "Image and mask filenames do not match"

# Split data into training and validation sets
train_images, valid_images, train_masks, valid_masks = train_test_split(
    image_files, mask_files, test_size=0.1, random_state=42
)

# Helper function to copy files
def copy_files(file_list, src_dir, dest_dir):
    for file_name in file_list:
        shutil.copy(os.path.join(src_dir, file_name), os.path.join(dest_dir, file_name))

# Copy files to respective directories
copy_files(train_images, image_dir, output_dirs['train_images'])
copy_files(train_masks, mask_dir, output_dirs['train_masks'])
copy_files(valid_images, image_dir, output_dirs['valid_images'])
copy_files(valid_masks, mask_dir, output_dirs['valid_masks'])

print("Data split and copied successfully!")