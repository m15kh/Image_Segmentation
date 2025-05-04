import torch
import yaml
import cv2
import os
import time

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES
from model import prepare_model

# Load configuration from YAML file
config_path = '/home/ubuntu/m15kh/Image_Segmentation/FineTune_DeepLabV3/params/inference.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Extract parameters from config
INPUT_DIR = config['input_dir']
OUTPUT_DIR = config['output_dir']
MODEL_PATH = config['model_path']
MAX_IMAGE_SIZE = config['max_image_size']

print(f"Configuration loaded from {config_path}")
print(f"Input directory: {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model path: {MODEL_PATH}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = prepare_model(len(ALL_CLASSES))
ckpt = torch.load(MODEL_PATH)
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

total_inference_time = 0  # Initialize total inference time

all_image_paths = os.listdir(INPUT_DIR)
for i, image_path in enumerate(all_image_paths):
    print(f"Processing Image {i+1}: {image_path}")
    image = Image.open(os.path.join(INPUT_DIR, image_path))

    if image.size[0] > MAX_IMAGE_SIZE:
        image = image.resize((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    
    image = image.convert('RGB')

    # Start timing
    start_time = time.time()

    outputs = get_segment_labels(image, model, device)
    outputs = outputs['out']
    segmented_image = draw_segmentation_map(outputs)

    # End timing
    end_time = time.time()
    inference_time = end_time - start_time
    total_inference_time += inference_time  # Accumulate inference time
    print(f"Inference Time for Image {i+1}: {inference_time:.4f} seconds")
    
    # Save the mask
    mask_path = os.path.join(OUTPUT_DIR, f"{image_path}")
    cv2.imwrite(mask_path, segmented_image)

    # Save the blended image
    final_image = image_overlay(image, segmented_image)
    blended_path = os.path.join(OUTPUT_DIR, f"blended_{image_path}")
    cv2.imwrite(blended_path, final_image)

print(f"Total Inference Time: {total_inference_time:.4f} seconds")  # Print total inference time
