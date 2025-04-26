from PIL import Image, ImageDraw
import json
import numpy as np

# Load the JSON file
json_path = "/home/rteam2/m15kh/Auto_Encoder/U_Net/data/30331320240684_porg_0.json"
with open(json_path, "r") as f:
    data = json.load(f)

# Image dimensions (update based on your image size)
image_width = 2448  # Replace with actual width
image_height = 2448  # Replace with actual height

# Create a blank mask for "e" labels
mask = np.zeros((image_height, image_width), dtype=np.uint8)  # Start with all zeros (black background)

# Extract "e" shapes and draw them on the mask
for shape in data["shapes"]:
    if shape["label"] == "e":
        # Ensure the points are in the correct format
        polygon = [(int(x), int(y)) for x, y in shape["points"]]
        img = Image.new("L", (image_width, image_height), 0)  # Start with black (0)
        ImageDraw.Draw(img).polygon(polygon, outline=255, fill=255)  # Fill the polygon with white (255)
        mask = np.maximum(mask, np.array(img))  # Combine the polygon area with the existing mask

# Save the mask as an image
mask_image = Image.fromarray(mask)  # No need to scale, already in 0-255 range
mask_image.save("e_mask.png")
