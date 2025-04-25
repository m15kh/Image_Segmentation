from PIL import Image, ImageDraw
import json
import numpy as np
import os

# Load the JSON file
json_path = "/home/rteam2/m15kh/Auto_Encoder/U_Net/data/30331320240684_porg_0.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Image dimensions (update based on your image size)
image_width = 2448  # Replace with actual width
image_height = 2448  # Replace with actual height

# Create a blank mask for "e" labels
mask_e = np.zeros((image_height, image_width), dtype=np.uint8)  # Start with all zeros (black background)

# Create a blank mask for "h" labels
mask_h = np.zeros((image_height, image_width), dtype=np.uint8)  # Start with all zeros (black background)

# Create a blank mask for "h" labels and invert it
mask_inverted = np.zeros((image_height, image_width), dtype=np.uint8)  # Start with all zeros (black background)

# Extract shapes and draw them on the respective masks
for shape in data["shapes"]:
    label = shape["label"]
    polygon = [(int(x), int(y)) for x, y in shape["points"]]  # Ensure the points are in the correct format
    img = Image.new("L", (image_width, image_height), 0)  # Start with black (0)
    ImageDraw.Draw(img).polygon(polygon, outline=255, fill=255)  # Fill the polygon with white (255)

    if label == "e":
        mask_e = np.maximum(mask_e, np.array(img))  # Combine the polygon area with the existing mask for "e"
    elif label == "h":
        mask_h = np.maximum(mask_h, np.array(img))  # Combine the polygon area with the existing mask for "h"

# Extract "h" shapes and draw them on the mask
for shape in data["shapes"]:
    if shape["label"] == "h":
        # Ensure the points are in the correct format
        polygon = [(int(x), int(y)) for x, y in shape["points"]]
        img = Image.new("L", (image_width, image_height), 0)  # Start with black (0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)  # Fill the polygon with white (1)
        mask_inverted |= np.array(img)  # Combine the polygon area

# Invert the mask colors
mask_inverted = 1 - mask_inverted  # Invert: 1 becomes 0, and 0 becomes 1

# Save the masks as images
mask_e_image = Image.fromarray(mask_e)  # No need to scale, already in 0-255 range
# mask_e_image.save("e_mask.png")

# mask_h_image = Image.fromarray(mask_h)  # No need to scale, already in 0-255 range
# mask_h_image.save("h_mask.png")

# Save the inverted mask as an image
mask_inverted_image = Image.fromarray(mask_inverted * 255)  # Scale to 0-255 for visualization
# mask_inverted_image.save("h_mask_inverted.png")

# Merge mask_inverted and mask_e
merged_mask = np.maximum(mask_inverted * 255, mask_e)  # Scale mask_inverted to 0-255 before merging

# Save the merged mask as an image
merged_mask_image = Image.fromarray(merged_mask)
# Save the merged mask using the name of the JSON file
json_filename = os.path.basename(json_path)  # Extract the filename from the path
mask_filename = os.path.splitext(json_filename)[0] + "_merged_mask.png"  # Replace extension with "_merged_mask.png"
merged_mask_image.save(mask_filename)