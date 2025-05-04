import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/m15kh/Desktop/project/fingerprint/Unet_fingerprint/source_code_data/inference_results/1024_1024_inference/mask_data/30331320240684_porg_0.png')  # Replace with your image file name

image = cv2.bitwise_not(image, image)
kernel = np.ones((5, 5), np.uint8)  # You can adjust the size

dilated = cv2.erode(image, kernel, iterations=1)

comparison = np.hstack((image, dilated))

# Save the result
cv2.imwrite('comparison_output.jpg', comparison)