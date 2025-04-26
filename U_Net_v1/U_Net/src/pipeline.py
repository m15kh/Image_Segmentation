
from torch.utils.data import DataLoader
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from U_Net.models.unet import UNet
from U_Net.models.dataloader import SegmentationDataset
from U_Net.models.train import train_unet

# Define paths for training and validation datasets
train_image_dir = '/home/rteam2/m15kh/Auto_Encoder/U_Net/data/images/'
train_mask_dir = '/home/rteam2/m15kh/Auto_Encoder/U_Net/data/masks/'

# Create Dataset and DataLoader for training and validation
train_dataset = SegmentationDataset(train_image_dir, train_mask_dir)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform=None)
# val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize the U-Net model
model = UNet(in_channels=3, out_channels=1)  # RGB input and binary output mask

# Train the U-Net model
train_unet(model, train_loader, val_loader=None, epochs=50, lr=1e-4, device='cuda')

# Save the trained model checkpoint
torch.save(model.state_dict(), "/home/rteam2/m15kh/Auto_Encoder/U_Net/checkpoints/unet_checkpoint.pth")

# Inference on a new image (example)
# test_image = Image.open("path/to/test/image.jpg").convert('RGB')
# test_image = transform(test_image)  # Apply same transformations as for training
# test_image = test_image.unsqueeze(0)  # Add batch dimension

# output_mask = inference(model, test_image, device='cuda')
