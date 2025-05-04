import os
from pathlib import Path

def check_image_mask_pairs(images_dir, masks_dir):
    """
    Check if each image in images_dir has a corresponding mask in masks_dir.
    
    Args:
        images_dir (str): Path to directory containing images
        masks_dir (str): Path to directory containing masks
        
    Returns:
        dict: Dictionary with matched and unmatched pairs
    """
    # Get all files in both directories
    image_files = os.listdir(images_dir)
    mask_files = os.listdir(masks_dir)
    
    # Extract filenames without extensions
    image_names = {Path(f).stem for f in image_files}
    mask_names = {Path(f).stem for f in mask_files}
    
    # Find matching and non-matching pairs
    matched_pairs = image_names.intersection(mask_names)
    images_without_masks = image_names - mask_names
    masks_without_images = mask_names - image_names
    
    result = {
        'matched_pairs': matched_pairs,
        'images_without_masks': images_without_masks,
        'masks_without_images': masks_without_images
    }
    
    return result

def print_report(result):
    """Print a report of the pair checking results"""
    print(f"Found {len(result['matched_pairs'])} matching pairs")
    
    if result['images_without_masks']:
        print(f"\n{len(result['images_without_masks'])} images don't have matching masks:")
        for img in sorted(result['images_without_masks']):
            print(f"  - {img}")
    
    if result['masks_without_images']:
        print(f"\n{len(result['masks_without_images'])} masks don't have matching images:")
        for mask in sorted(result['masks_without_images']):
            print(f"  - {mask}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check if image-mask pairs exist')
    parser.add_argument('--images', default='/home/rteam2/m15kh/FPR_dataset/Finger_Data/inference_test/images', help='Directory containing images')
    parser.add_argument('--masks', default='/home/rteam2/m15kh/FPR_dataset/Finger_Data/inference_test/masks', help='Directory containing masks')
    args = parser.parse_args()
    
    if not os.path.isdir(args.images):
        print(f"Error: Images directory '{args.images}' does not exist")
        exit(1)
    
    if not os.path.isdir(args.masks):
        print(f"Error: Masks directory '{args.masks}' does not exist")
        exit(1)
    
    result = check_image_mask_pairs(args.images, args.masks)
    print_report(result)