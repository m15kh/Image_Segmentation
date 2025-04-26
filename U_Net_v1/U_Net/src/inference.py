import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from U_Net.models.unet import UNet

def load_model(checkpoint_path, in_channels=3, out_channels=1, device='cuda'):
    """
    Load a trained UNet model from a checkpoint file
    """
    model = UNet(in_channels=in_channels, out_channels=out_channels)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess an input image for inference
    """
    # Open image and convert to RGB
    image = Image.open(image_path).convert('RGB')
    
    # Resize to target size
    image = image.resize(target_size)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image)
    return tensor.unsqueeze(0), image  # Add batch dimension

def perform_inference(model, image_tensor, device='cuda'):
    """
    Perform inference on a preprocessed image tensor
    """
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(output)
    
    return pred

def postprocess_prediction(prediction, threshold=0.5):
    """
    Convert prediction tensor to binary mask
    """
    # Convert to numpy
    pred_np = prediction.squeeze().cpu().numpy()
    
    # Apply threshold to get binary mask
    binary_mask = (pred_np > threshold).astype(np.uint8) * 255
    
    return binary_mask

def visualize_results(original_image, mask, save_path=None):
    """
    Create visualization of the original image and predicted mask and save it
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the figure to avoid display

def main():
    # Set paths
    checkpoint_path = "/home/rteam2/m15kh/Auto_Encoder/U_Net/checkpoints/unet_checkpoint.pth"
    test_image_path = "/home/rteam2/m15kh/Auto_Encoder/U_Net/data/test_data/30331320240684_porg_0.png"  # Replace with your test image
    output_dir = "/home/rteam2/m15kh/Auto_Encoder/U_Net/results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(checkpoint_path)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(test_image_path)
    
    # Perform inference
    prediction = perform_inference(model, image_tensor)
    
    # Postprocess prediction
    binary_mask = postprocess_prediction(prediction)
    
    # Save visualization without displaying
    result_path = os.path.join(output_dir, os.path.basename(test_image_path).split('.')[0] + "_result.png")
    visualize_results(original_image, binary_mask, save_path=result_path)
    
    # Save binary mask
    mask_path = os.path.join(output_dir, os.path.basename(test_image_path).split('.')[0] + "_mask.png")
    Image.fromarray(binary_mask).save(mask_path)
    
    print(f"Results saved to {result_path}")
    print(f"Mask saved to {mask_path}")

if __name__ == "__main__":
    main()
