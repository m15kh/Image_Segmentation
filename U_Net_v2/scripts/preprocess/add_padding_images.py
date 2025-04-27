import os
import numpy as np
from PIL import Image
import cv2

# Function to add white padding to make all images consistent size
def add_white_padding(image, target_width, target_height):
    # Get current image dimensions
    if len(image.shape) == 3:
        current_height, current_width, channels = image.shape
    else:
        current_height, current_width = image.shape
        channels = 1
    
    # If the image is already the target size, return it unchanged
    if current_width == target_width and current_height == target_height:
        return image
    
    # Create a new white canvas of the target size
    if channels == 3:
        padded_image = np.ones((target_height, target_width, channels), dtype=np.uint8) * 255
    else:
        padded_image = np.ones((target_height, target_width), dtype=np.uint8) * 255
    
    # Calculate padding
    if current_width < target_width:
        # Add padding on left and right equally
        left_padding = (target_width - current_width) // 2
        
        # Place the original image in the center of the padded image
        if channels == 3:
            padded_image[:current_height, left_padding:left_padding+current_width, :] = image
        else:
            padded_image[:current_height, left_padding:left_padding+current_width] = image
    else:
        # If no width padding needed, just handle height
        top_padding = (target_height - current_height) // 2 if current_height < target_height else 0
        if channels == 3:
            padded_image[top_padding:top_padding+current_height, :current_width, :] = image
        else:
            padded_image[top_padding:top_padding+current_height, :current_width] = image
        
    return padded_image

# Process all images in an input folder
def process_all_images(input_folder, output_folder, target_width=2400, target_height=2600):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all image files in the input folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(input_folder) 
                   if os.path.isfile(os.path.join(input_folder, f)) and 
                   any(f.lower().endswith(ext) for ext in image_extensions)]

    # Process each image file
    for image_file in image_files:
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        
        # Read the image
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"Failed to load image: {input_path}")
            continue
            
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Add padding if necessary to reach target size
        if width < target_width or height < target_height:
            padded_image = add_white_padding(image, target_width, target_height)
            print(f"Applied white padding to image from {width}x{height} to {target_width}x{target_height}")
        else:
            padded_image = image
            
        # Save the padded image
        cv2.imwrite(output_path, padded_image)
        print(f"Saved padded image: {output_path}")

# Define your input folder and output folder
input_folder = "/home/rteam2/m15kh/FPR_dataset/pure_images"
output_folder = "/home/rteam2/m15kh/FPR_dataset/images"

# Define your target image dimensions
target_width = 2400  # Target width for all images
target_height = 2600  # Target height for all images

# Process all images in the input folder
process_all_images(input_folder, output_folder, target_width, target_height)