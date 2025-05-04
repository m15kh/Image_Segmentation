import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Function to get image dimensions from JSON file
def get_image_dimensions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Check if the imageWidth and imageHeight are specified in the JSON
    if "imageWidth" in data and "imageHeight" in data:
        return data["imageWidth"], data["imageHeight"]
    else:
        # Default to the larger size if not specified
        return 2400, 2600

# Function to add padding to make all images consistent size
def add_padding(mask, target_width, target_height):
    current_height, current_width = mask.shape
    
    # If the image is already the target size, return it unchanged
    if current_width == target_width and current_height == target_height:
        return mask
    
    # Create a new black canvas of the target size
    padded_mask = np.zeros((target_height, target_width), dtype=np.uint8)
    
    # Calculate padding
    if current_width < target_width:
        # Add padding on left and right equally
        left_padding = (target_width - current_width) // 2
        
        # Place the original mask in the center of the padded mask
        padded_mask[:current_height, left_padding:left_padding+current_width] = mask
    else:
        # If no width padding needed, just handle height
        top_padding = (target_height - current_height) // 2 if current_height < target_height else 0
        padded_mask[top_padding:top_padding+current_height, :current_width] = mask
        
    return padded_mask

# Function to process each JSON file
def process_json(json_path, output_dir, target_width=2400, target_height=2600):
    # Get actual image dimensions from the JSON file
    image_width, image_height = get_image_dimensions(json_path)
    
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create blank masks with the actual image dimensions
    mask_e = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_h = np.zeros((image_height, image_width), dtype=np.uint8)
    mask_inverted = np.zeros((image_height, image_width), dtype=np.uint8)

    # Extract shapes and draw them on the respective masks
    for shape in data["shapes"]:
        label = shape["label"]
        polygon = [(int(x), int(y)) for x, y in shape["points"]]
        img = Image.new("L", (image_width, image_height), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=255, fill=255)

        if label == "e":
            mask_e = np.maximum(mask_e, np.array(img))
        elif label == "h":
            mask_h = np.maximum(mask_h, np.array(img))

    # Extract and process "h" shapes for inverted mask
    for shape in data["shapes"]:
        if shape["label"] == "h":
            polygon = [(int(x), int(y)) for x, y in shape["points"]]
            img = Image.new("L", (image_width, image_height), 0)
            ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
            mask_inverted |= np.array(img)

    # Invert the mask colors
    mask_inverted = 1 - mask_inverted

    # Merge the inverted mask and mask_e
    merged_mask = np.maximum(mask_inverted * 255, mask_e)
    merged_mask = cv2.bitwise_not(merged_mask)
    
    # Add padding if necessary to reach target size
    if image_width < target_width or image_height < target_height:
        merged_mask = add_padding(merged_mask, target_width, target_height)
        print(f"Applied padding to mask from {image_width}x{image_height} to {target_width}x{target_height}")
    
    # Create output path for the merged mask image
    json_filename = os.path.basename(json_path)
    mask_filename = os.path.splitext(json_filename)[0] + ".png"
    mask_path = os.path.join(output_dir, mask_filename)

    # Save the merged mask as an image
    merged_mask_image = Image.fromarray(merged_mask)
    merged_mask_image.save(mask_path)
    print(f"Saved mask: {mask_path}")

# Main function to process all JSON files in an input folder
def process_all_json(input_folder, output_folder, target_width=2400, target_height=2600):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all JSON files in the input folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(input_folder, json_file)
        process_json(json_path, output_folder, target_width, target_height)

# Define your input folder and output folder
input_folder = "/home/rteam2/m15kh/FPR_dataset/label_json"
output_folder = "/home/rteam2/m15kh/FPR_dataset/masks_b"

# Define your target image dimensions
target_width = 2400  # Target width for all masks
target_height = 2600  # Target height for all masks

# Process all JSON files in the input folder
process_all_json(input_folder, output_folder, target_width, target_height)