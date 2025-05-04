from empatches import EMPatches
import imgviz  # just for patch visualization
import cv2
import numpy as np

# Read and convert the image
img = cv2.imread('/home/m15kh/Desktop/project/fingerprint/Unet_fingerprint/source_code_data/30331320240684_porg_0.png')


patch_height = img.shape[0] // 4
patch_width = img.shape[1] // 4

# Extract patches
emp = EMPatches()
img_patches, indices = emp.extract_patches(img, patchsize=1600, overlap=0)

# Create tiled visualization
tiled = imgviz.tile(list(map(np.uint8, img_patches)), border=(255, 0, 0))

# Convert to BGR for OpenCV saving

# Save the tiled image
cv2.imwrite("tiled_patches_output_cv.png", tiled)
print("Saved tiled image using OpenCV.")