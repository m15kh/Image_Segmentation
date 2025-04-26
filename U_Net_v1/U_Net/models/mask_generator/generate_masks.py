import os
import json
import numpy as np
from PIL import Image, ImageDraw

# Function to process each JSON file
def process_json(json_path, output_dir, image_width, image_height):
    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create blank masks
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

    # Create output path for the merged mask image
    json_filename = os.path.basename(json_path)
    mask_filename = os.path.splitext(json_filename)[0] + "_mask.png"
    mask_path = os.path.join(output_dir, mask_filename)

    # Save the merged mask as an image
    merged_mask_image = Image.fromarray(merged_mask)
    merged_mask_image.save(mask_path)
    print(f"Saved mask: {mask_path}")

# Main function to process all JSON files in an input folder
def process_all_json(input_folder, output_folder, image_width, image_height):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all JSON files in the input folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

    # Process each JSON file
    for json_file in json_files:
        json_path = os.path.join(input_folder, json_file)
        process_json(json_path, output_folder, image_width, image_height)

# Define your input folder and output folder
input_folder = "/home/rteam2/m15kh/Auto_Encoder/U_Net/data/label"
output_folder = "/home/rteam2/m15kh/Auto_Encoder/U_Net/data/mask"

# Define your image dimensions
image_width = 2448  # Replace with actual width
image_height = 2448  # Replace with actual height

# Process all JSON files in the input folder
process_all_json(input_folder, output_folder, image_width, image_height)
