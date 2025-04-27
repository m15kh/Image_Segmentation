

import os
import glob

# Path to data folder
data_folder = "/home/ubuntu/m15kh/Image_Segmentation/FineTune_SynDiff/Finger_Data/Dataset/masks"


# Option 2: Remove specific list of images
# images_to_remove = ["image1.png", "image2.png", "image3.png"]
images_to_remove = ["61222720240243_porg_2.png"]

for img in images_to_remove:
    file_path = os.path.join(data_folder, img)
    if os.path.exists(file_path):
        print(f"Removing {file_path}")
        os.remove(file_path)
    else:
        print(f"File not found: {file_path}")