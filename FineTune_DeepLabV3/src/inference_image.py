import torch
import argparse
import cv2
import os
import time  # Import the time module

from utils import get_segment_labels, draw_segmentation_map, image_overlay
from PIL import Image
from config import ALL_CLASSES
from model import prepare_model

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', 
    help='path to input dir', 
    default='/home/ubuntu/m15kh/Image_Segmentation/Finger_Data/inference_test'
)
args = parser.parse_args()

out_dir = os.path.join('/home/ubuntu/m15kh/Image_Segmentation/FineTune_DeepLabV3', 'outputs', 'inference_results')
os.makedirs(out_dir, exist_ok=True)

# Set computation device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = prepare_model(len(ALL_CLASSES))
ckpt = torch.load('/home/ubuntu/m15kh/Image_Segmentation/best_model_finetuen_deeplabv3_2025)3)27.pth')
model.load_state_dict(ckpt['model_state_dict'])
model.eval().to(device)

total_inference_time = 0  # Initialize total inference time

all_image_paths = os.listdir(args.input)
for i, image_path in enumerate(all_image_paths):
    print(f"Processing Image {i+1}: {image_path}")
    image = Image.open(os.path.join(args.input, image_path))

    if image.size[0] > 1024:
        image = image.resize((1024, 1024))
    
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
    mask_path = os.path.join(out_dir, f"mask_{image_path}")
    cv2.imwrite(mask_path, segmented_image)

    # Save the blended image
    final_image = image_overlay(image, segmented_image)
    blended_path = os.path.join(out_dir, f"blended_{image_path}")
    cv2.imwrite(blended_path, final_image)

print(f"Total Inference Time: {total_inference_time:.4f} seconds")  # Print total inference time
