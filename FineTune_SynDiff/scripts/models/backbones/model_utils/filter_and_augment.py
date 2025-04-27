import os
import cv2
import math
import tqdm
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


class Augmentation:


    def process_img(self, img, gap = 170, err = 10, linewidth = 20):
    
        # initiation data/var
        draw = ImageDraw.Draw(img) 
        width, height = img.size

        class Coord:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        currx = 0
        curry = 0
        coordlist = []

        # generate the set of points
        while currx <= width:
            while curry <= height:
                coordlist.append(Coord( \
                currx + random.randint(0,err), \
                curry + random.randint(0,err) \
                ))        
                curry += gap
            curry = gap
            currx += gap

        # calculate endpoint with angle/length
        def calcEnd(x, y, angle, length):
            endx = int(x - (math.cos(math.radians(angle)) * length))
            endy = int(y - (math.sin(math.radians(angle)) * length))
            return endx, endy

        # draw line with random angle/length
        for c in coordlist:
            length = random.randint(10, 50)
            randangle = random.randint(0,359)
            endx, endy = calcEnd(c.x, c.y, randangle, length)
            draw.line((c.x, c.y, endx, endy), fill=random.randint(0,255), width=linewidth)

        img.convert('L')

        
        return img


    def add_noise(self, img): 
    
        # Getting the dimensions of the image 
        row , col = img.shape 
        
        # Randomly pick some pixels in the 
        # image for coloring them white 
        # Pick a random number between 300 and 10000 
        number_of_pixels = random.randint(300, 10000) 
        for i in range(number_of_pixels): 
            
            # Pick a random y coordinate 
            y_coord=random.randint(0, row - 1) 
            
            # Pick a random x coordinate 
            x_coord=random.randint(0, col - 1) 
            
            # Color that pixel to white 
            img[y_coord][x_coord] = 255
            
        # Randomly pick some pixels in 
        # the image for coloring them black 
        # Pick a random number between 300 and 10000 
        number_of_pixels = random.randint(300 , 10000) 
        for i in range(number_of_pixels): 
            
            # Pick a random y coordinate 
            y_coord=random.randint(0, row - 1) 
            
            # Pick a random x coordinate 
            x_coord=random.randint(0, col - 1) 
            
            # Color that pixel to black 
            img[y_coord][x_coord] = 0
            
        return img 


    def add_speckle_noise(self, image, std_dev=0.1):
        """
        Add speckle noise to the image.
        """
        h, w = image.shape
        noise = np.random.normal(0, std_dev, (h, w))
        noisy_image = np.clip(image + image * noise, 0, 255).astype(np.uint8)
        return noisy_image


    def add_poisson_noise(self, image):
        """
        Add Poisson noise to the image.
        """
        # Generate noise from a Poisson distribution with the same shape as the image
        noise = np.random.poisson(image, image.shape)
        # Clip the noisy image to the valid range [0, 255]
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def draw_random_lines(self, image, num_lines=10, line_thickness=1):
        """
        Draw random lines on the image.
        """
        h, w = image.shape[:2]

        # Create a copy of the image to draw lines on
        image_with_lines = image.copy()

        # Generate random endpoints for each line
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
            x2, y2 = np.random.randint(0, w), np.random.randint(0, h)

            # Choose a random color for the line
            color = np.random.randint(150, 256, (3,)).tolist()

            # Draw the line on the image
            cv2.line(image_with_lines, (x1, y1), (x2, y2), color, line_thickness)

        return image_with_lines


    def non_uniform_illumination(self, image):
        rows, cols = image.shape
        gradient = np.linspace(1, 0.5, cols)
        illumination_mask = np.tile(gradient, (rows, 1)).astype(np.float32)
        illuminated_image = image.astype(np.float32) * illumination_mask
        return cv2.normalize(illuminated_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    def perspective_transformation(self, image, mask, ori):
        rows, cols = image.shape
        src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
        dst_points = np.float32([[0, rows*0.1], [cols-1, 0], [0, rows*0.9], [cols-1, rows-1]])
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.warpPerspective(image, M, (cols, rows))
        mask = cv2.warpPerspective(mask, M, (cols, rows))
        ori = cv2.warpPerspective(ori, M, (cols, rows))
        return image, mask, ori

    def motion_blur(self, image, size=15):
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        return cv2.filter2D(image, -1, kernel_motion_blur)


    def adversarial_augmentation(self, image, epsilon=0.01):
        noise = np.random.randn(*image.shape) * epsilon * 255
        adv_image = image.astype(np.float32) + noise
        return np.clip(adv_image, 0, 255).astype(np.uint8)

    def simulated_environmental_effects(self, image, num_of_dust = 50):
        overlay = image.copy()
        rows, cols = image.shape

        # Simulate dust particles
        for _ in range(num_of_dust):  # Adjust the number of particles
            x, y = np.random.randint(0, cols), np.random.randint(0, rows)
            cv2.circle(overlay, (x, y), np.random.randint(1, 5), (255, 255, 255), -1)
        
        # Simulate water droplets
        for _ in range(20):  # Adjust the number of droplets
            x, y = np.random.randint(0, cols), np.random.randint(0, rows)
            radius = np.random.randint(5, 15)
            x1, y1 = max(0, x-radius), max(0, y-radius)
            x2, y2 = min(cols, x+radius), min(rows, y+radius)
            cv2.circle(overlay, (x, y), radius, (255, 255, 255), -1)
            overlay[y1:y2, x1:x2] = cv2.GaussianBlur(overlay[y1:y2, x1:x2], (5, 5), 0)

        alpha = random.uniform(0.5 , 0.8)  # Transparency factor
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


        

    def do_augment(self, nfiq_path = None, apply_nfiq_on_masks = True, use_mpoints = True):
        
        root = "enahancer_training_data_temp/masks"
        if nfiq_path != None:
            database = pd.read_csv(nfiq_path)
            paths = database['Filename'].tolist()
            qs = database['QualityScore'].tolist()

        else:
            paths = glob.glob(root + "/*") 


        for i in tqdm.tqdm(range(len(paths)),
                           desc='Augmenting Images',
                            unit='Image',
                            ncols = 100,
                            total = len(paths)):                
            image_path           = os.path.join(root.replace("masks", "images"), paths[i].split(os.path.sep)[-1])
            mask_path            = os.path.join(root, paths[i].split(os.path.sep)[-1])
            if use_mpoints:
                mpoints_path         = os.path.join(root.replace("masks", "mpoints"), paths[i].split(os.path.sep)[-1])
            orientations_path    = os.path.join(root.replace("masks", "orientations"), paths[i].split(os.path.sep)[-1])
            quality = qs[i] if nfiq_path != None else None
        
            try:
                if quality == None or (nfiq_path != None and int(quality) > 50):
                        
                    
                    name = image_path.split(os.path.sep)[-1].split(".png")[0]
                    image        = Image.open(image_path)
                    mask         = Image.open(mask_path) 
                    if use_mpoints:               
                        mpoints      = Image.open(mpoints_path)
                    orientations = Image.open(orientations_path)

                    image         = image.convert("RGB")
                    mask          = mask.convert("RGB")
                    if use_mpoints:
                        mpoints       = mpoints.convert("RGB")
                    orientations  = orientations.convert("RGB")
                                                            

                    image_line = self.process_img(image.copy(), gap = random.randint(50, 200), err = random.randint(1, 20), linewidth = random.randint(5, 30))
                    image_line = np.expand_dims(cv2.cvtColor(np.array(image_line), cv2.COLOR_RGB2GRAY), axis = -1)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_line_{name}.png", image_line)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_line_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY), axis = -1))
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_line_{name}.png", np.expand_dims(cv2.cvtColor(np.array(orientations), cv2.COLOR_RGB2GRAY), axis = -1))  
                    if use_mpoints:
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_line_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))

            
                    
                    #APPLY AUGMENTATION -> Change Background Color
                    background_color = np.ones(shape = np.array(image).shape) * random.randint(30, 180)
                    image_background = cv2.addWeighted(np.array(image).astype(np.uint8), 0.1, background_color.astype(np.uint8), 0.9, 0) 
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_background_{name}.png", image_background)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_background_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY), axis = -1))
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_background_{name}.png", np.expand_dims(cv2.cvtColor(np.array(orientations), cv2.COLOR_RGB2GRAY), axis = -1))  
                    if use_mpoints:
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_background_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))

            
                    
                    
                    #APPLY AUGMENTATION -> Add noise
                    image_gau               = cv2.GaussianBlur(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.uint8), (7, 7), 0) 
                    image_speckle           = self.add_speckle_noise(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.uint8), std_dev=random.uniform(0.1, 0.2))
                    image_non_uniform_illum = self.non_uniform_illumination(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY))
                    image_perspective_trans, mask_perspective_trans, ori_perspective_trans = self.perspective_transformation(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.uint8), 
                                                                                                                        cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY).astype(np.uint8), 
                                                                                                                        np.array(orientations))
                    image_adv_augmented     = self.adversarial_augmentation(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.uint8), epsilon = random.uniform(0.05,0.3))
                    image_env_effects       = self.simulated_environmental_effects(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.uint8), num_of_dust = random.randint(40, 80))                                    
                    image_image_line_draw   = self.draw_random_lines(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY).astype(np.uint8), num_lines=random.randint(1, 10), line_thickness=random.randint(2, 5))
                    
                    
                    
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_non_uniform_illum_{name}.png", image_non_uniform_illum)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_non_uniform_illum_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY), axis = -1))
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_non_uniform_illum_{name}.png", np.expand_dims(cv2.cvtColor(np.array(orientations), cv2.COLOR_RGB2GRAY), axis = -1))   
                    if use_mpoints:
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_non_uniform_illum_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))

                        
                    
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_perspective_trans_{name}.png", image_perspective_trans)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_perspective_trans_{name}.png", mask_perspective_trans)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_perspective_trans_{name}.png", ori_perspective_trans)  
                    if use_mpoints:                        
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_perspective_trans_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))
                    
                        

                    
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_adv_augmented_{name}.png", image_adv_augmented)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_adv_augmented_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY), axis = -1))
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_adv_augmented_{name}.png", np.expand_dims(cv2.cvtColor(np.array(orientations), cv2.COLOR_RGB2GRAY), axis = -1))    
                    if use_mpoints:
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_adv_augmented_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))

                        
                            
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_env_effects_{name}.png", image_env_effects)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_env_effects_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY), axis = -1))
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_env_effects_{name}.png", np.expand_dims(cv2.cvtColor(np.array(orientations), cv2.COLOR_RGB2GRAY), axis = -1))  
                    if use_mpoints:
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_env_effects_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))

            
                                                                            
                    
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_gau_{name}.png", image_gau)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_gau_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY), axis = -1))
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_gau_{name}.png", np.expand_dims(cv2.cvtColor(np.array(orientations), cv2.COLOR_RGB2GRAY), axis = -1))  
                    if use_mpoints:
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_gau_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))

            
                    
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_speckle_{name}.png", image_speckle)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_speckle_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY), axis = -1))
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_speckle_{name}.png", np.expand_dims(cv2.cvtColor(np.array(orientations), cv2.COLOR_RGB2GRAY), axis = -1))
                    if use_mpoints:
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_speckle_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))

            
       
                    
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/images/image_image_line_draw_{name}.png", image_image_line_draw)
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/masks/image_image_line_draw_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2GRAY), axis = -1))
                    cv2.imwrite(f"{root.split(os.path.sep)[0]}/orientations/image_image_line_draw_{name}.png", np.expand_dims(cv2.cvtColor(np.array(orientations), cv2.COLOR_RGB2GRAY), axis = -1))  
                    if use_mpoints:
                        cv2.imwrite(f"{root.split(os.path.sep)[0]}/mpoints/image_image_line_draw_{name}.png", np.expand_dims(cv2.cvtColor(np.array(mpoints), cv2.COLOR_RGB2GRAY), axis = -1))

                
                                    
            except:
                pass
            
            
