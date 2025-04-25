import os
import sys
from typing import List, Tuple, Union
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import cv2
import time
import torch
import time
import yaml
from argparse import Namespace
from scripts.models.net import BuildUnet
from scripts.models.modules import InferenceBaseModule
from torch.cuda.amp import autocast

# Add import for EMPatches
from empatches import EMPatches

class SegmentationInference(InferenceBaseModule):
    def __init__(self, checkpoint_path: str, batch_size: int = 1): 
        self.checkpoint_path = checkpoint_path
        assert os.path.isfile(self.checkpoint_path), f"{__file__} checkpoint path is not valid"
        self.init_model()
        
        self.num_gpus = torch.cuda.device_count()
        if batch_size <= 0:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

    def init_model(self):
        self.model = BuildUnet()
        checkpoint = torch.load(self.checkpoint_path, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print("-------------------------------------")
        print(checkpoint['width'])
        print(checkpoint['height'])
        # print(checkpoint['state_dict'])
        print("-------------------------------------")
        self.width = checkpoint["width"]
        self.height = checkpoint["height"]
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

    def _patch_image(self, image: np.array, patch_size: int, overlap: float) -> Tuple[List[Tuple[np.array, Tuple[int, int, int, int]]], Tuple[int, int]]:
        """
        Split an image into overlapping patches.
        
        Args:
            image (np.array): Input image
            patch_size (int): Size of each patch (square)
            overlap (float): Overlap between patches (0-1)
        
        Returns:
            List[Tuple[np.array, Tuple[int, int, int, int]]]: List of patches and their coordinates
            Tuple[int, int]: Original image size
        """
        h, w = image.shape[:2]
        stride = int(patch_size * (1 - overlap))
        
        patches = []
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + patch_size, h)
                x2 = min(x + patch_size, w)
                
                # If we're at the border, adjust to take a full-sized patch
                if y2 - y < patch_size and y + patch_size <= h:
                    y = max(0, y2 - patch_size)
                    y2 = y + patch_size
                if x2 - x < patch_size and x + patch_size <= w:
                    x = max(0, x2 - patch_size)
                    x2 = x + patch_size
                    
                patch = image[y:y2, x:x2]
                
                # If patch is smaller than patch_size (at the borders), pad it
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    temp_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                    temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = temp_patch
                    
                patches.append((patch, (x, y, x2, y2)))
        
        return patches, (h, w)

    def _combine_patches(self, patches: List[Tuple[np.array, Tuple[int, int, int, int]]], img_size: Tuple[int, int]) -> np.array:
        """
        Combine patches back into a full image.
        
        Args:
            patches (List[Tuple[np.array, Tuple[int, int, int, int]]]): List of patches and their coordinates
            img_size (Tuple[int, int]): Original image size
        
        Returns:
            np.array: Combined image
        """
        h, w = img_size
        result = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        for mask, (x1, y1, x2, y2) in patches:
            # Create weight map for blending to avoid visible seams
            patch_h, patch_w = y2 - y1, x2 - x1
            patch_weight = np.ones((patch_h, patch_w), dtype=np.float32)
            
            # Resize the mask to match the patch size
            patch_mask = cv2.resize(mask, (patch_w, patch_h)) if mask.shape[:2] != (patch_h, patch_w) else mask
            
            # Add the weighted mask to the result
            result[y1:y2, x1:x2] += patch_mask * patch_weight
            weight_map[y1:y2, x1:x2] += patch_weight
        
        # Average the result where patches overlapped
        weight_map = np.maximum(weight_map, 1)
        result = result / weight_map
        
        return result

    def inference(self, image: Union[List[np.array], np.array], device_id: int = 0, patch_size: int = 1360, overlap: float = 0.1) -> list[np.array]:
        """input a list of RGB images, patch them, and process each patch

        Args:
            images (list[np.array]): list of images
            device_id (int): GPU device ID
            patch_size (int): Size of each patch
            overlap (float): Overlap between patches (0-1)

        Returns:
            list[np.array]: list of masks
        """
        
        device = f"cuda:{device_id}"
        self.model.to(device)
        
        results = []
        
        # Handle single image
        if isinstance(image, np.ndarray):
            image_list = [image]
        else:
            image_list = image
            
        for img in image_list:
            # Patch the image
            patches, img_size = self._patch_image(img, patch_size, overlap)
            patch_masks = []
            
            # Process patches in batches
            for i in range(0, len(patches), self.batch_size):
                batch_patches = [p[0] for p in patches[i:i+self.batch_size]]
                batch_tensor, batch_shape = self._preprocess(batch_patches)
                batch_tensor = batch_tensor.to(device)
                
                with torch.no_grad():
                    with autocast():
                        preds = self.model(batch_tensor)
                
                for j, (pred, patch_shape) in enumerate(zip(preds, batch_shape)):
                    try:
                        mask = self._postprocess(pred, *patch_shape)
                        patch_masks.append((mask, patches[i+j][1]))
                    except:
                        patch_masks.append((np.zeros(patch_shape[::-1], dtype=np.uint8), patches[i+j][1]))
            
            # Combine the patch predictions
            result_mask = self._combine_patches(patch_masks, img_size)
            results.append(result_mask)
        
        return results

    def batch_inference(self, image_dir: str, file_extention: str, output_path: str, device_id:int=0, patch_size:int=1360, overlap:float=0.1):
        """
        input image directory image
        write output mask to look for fingerprint

        Args:
            image_path (str): directory path
            image_extension (str): extention
            output_path (str): output path
            device_id (int): GPU device ID
            patch_size (int): Size of each patch
            overlap (float): Overlap between patches (0-1)
        """

        start_time = time.time()
        assert type(device_id)==int, "{__file__} device_id should be integer"
        os.makedirs(output_path, exist_ok=True)
        image_list = self._data_set(image_dir, file_extention)
        
        for batch_name, batch_images in self._data_loader(image_list,self.batch_size*self.num_gpus):
            batch_preds = self.inference(batch_images, device_id, patch_size, overlap)
            # postprocess
            for img_name, pred in zip(batch_name, batch_preds):
                if pred is None:
                    print(f"Skipping image {img_name} as prediction is None.")
                    continue  

                output_file_path = os.path.join(output_path, img_name)
                pred = np.array(pred)

                if pred.dtype != np.uint8:
                    pred = (pred * 255).astype(np.uint8)

                cv2.imwrite(output_file_path, pred)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Batch inference took {execution_time:.4f} seconds")

    def _postprocess(self, image:torch.Tensor, w:int, h:int):
        image = torch.sigmoid(image)
        image = torch.squeeze(image)
        image = image.cpu().numpy()
        image = (image > 0.5).astype(np.uint8)
        image = cv2.resize(image.copy(), (w, h))
        _, pred_mask_binary = cv2.threshold(image, 0.5, 255, cv2.THRESH_BINARY)
        
        contours, hierarchy = cv2.findContours(pred_mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(pred_mask_binary, [max_contour], -1, 255, thickness=cv2.FILLED)
        return pred_mask_binary
    
    def _transform(self, image:np.array):
        image = cv2.resize(image, (self.width, self.height))
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = np.stack(image)             
        image = torch.from_numpy(image)
        return image
    
    def _preprocess(self, images: Union[List[np.array], np.array]) -> torch.Tensor:
        if type(images) == np.ndarray:
            assert len(images.shape) == 3, f"{__file__} input image should be RGB but is gray"
            return self._transform(np.array(images)).unsqueeze(0), [images.shape[:2][::-1]]

        elif type(images) == list:
            batch_image = []
            batch_shape = []
            assert len(images) > 0, f"{__file__} input images is empty"

            for img in images:
                assert len(img.shape) == 3, f"{__file__} input image should be RGB but is gray"
                batch_shape.append(img.shape[:2][::-1])
                batch_image.append(self._transform(np.array(img)))
            
            return torch.stack(batch_image), batch_shape
        else:
            raise TypeError(f"{__file__} input image should be list or np.ndarray but is {type(images)}")

    def _data_set(self, image_dir:str, file_extention:str):
        if os.path.isdir(image_dir):
            out_list = [p for p in Path(image_dir).glob(f"**/*{file_extention}")]
        elif os.path.isfile(image_dir):
            out_list = [Path(image_dir)]
        else:
            raise ValueError(f"{__file__} image_path is not valid")
        assert len(out_list), f"{__file__} does not found any image in the directory"
        return out_list

    def _data_loader(self, ds:List, batch_size:int):
        batch_name = []
        batch_images = []
        for idx, p in enumerate(ds):
            image = cv2.imread(p.as_posix())
            batch_name.append(p.name)
            batch_images.append(image)
            if len(batch_images) % batch_size==0 or idx==(len(ds)-1):
                yield batch_name, batch_images
                batch_name = []
                batch_images = []

    def _patch_image_emp(self, image: np.array, patch_size: int = 1360, overlap: float = 0.1):
        """
        Split an image into overlapping patches using EMPatches.
        
        Args:
            image (np.array): Input image
            patch_size (int): Size of each patch (square)
            overlap (float): Overlap between patches (0-1)
        
        Returns:
            patches: List of image patches
            indices: Indices for reconstruction
        """
        emp = EMPatches()
        patches, indices = emp.extract_patches(image, patchsize=patch_size, overlap=overlap)
        return patches, indices

    def _merge_patches_emp(self, patches, indices, mode='avg'):
        """
        Merge patches back into a full image using EMPatches.
        
        Args:
            patches: List of image patches
            indices: Indices from extraction
            mode (str): Merging mode ('avg', 'max', etc.)
        
        Returns:
            np.array: Merged image
        """
        # Ensure all patches have the correct shape based on the indices
        for i, (patch, idx) in enumerate(zip(patches, indices)):
            y_start, y_end, x_start, x_end = idx
            required_shape = (y_end - y_start, x_end - x_start)
            
            if patch.shape[:2] != required_shape:
                # Reshape the patch to match the required dimensions
                patches[i] = cv2.resize(patch, (required_shape[1], required_shape[0]))
        
        emp = EMPatches()
        merged_img = emp.merge_patches(patches, indices, mode=mode)
        return merged_img

    def inference_with_emp_patches(self, image: Union[List[np.array], np.array], device_id: int = 0, 
                                patch_size: int = 1360, overlap: float = 0.1) -> list[np.array]:
        """
        Process images using EMPatches for patch extraction and merging.
        
        Args:
            images (list[np.array]): list of images
            device_id (int): GPU device ID
            patch_size (int): Size of each patch
            overlap (float): Overlap between patches (0-1)

        Returns:
            list[np.array]: list of masks
        """
        device = f"cuda:{device_id}"
        self.model.to(device)
        
        results = []
        
        # Handle single image
        if isinstance(image, np.ndarray):
            image_list = [image]
        else:
            image_list = image
            
        for img in image_list:
            # Patch the image using EMPatches
            patches, indices = self._patch_image_emp(img, patch_size, overlap)
            mask_patches = []
            
            # Process patches in batches
            for i in range(0, len(patches), self.batch_size):
                batch_patches = patches[i:i+self.batch_size]
                batch_tensor, batch_shape = self._preprocess(batch_patches)
                batch_tensor = batch_tensor.to(device)
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):  # Fix deprecated autocast warning
                        preds = self.model(batch_tensor)
                
                for j, (pred, patch_shape) in enumerate(zip(preds, batch_shape)):
                    try:
                        # Process the patch prediction
                        mask = self._postprocess(pred, *patch_shape)
                        
                        # Ensure the mask is the same size as the original patch
                        # This is critical for EMPatches to work correctly
                        patch_h, patch_w = patches[i+j].shape[:2]
                        if mask.shape[0] != patch_h or mask.shape[1] != patch_w:
                            mask = cv2.resize(mask, (patch_w, patch_h))
                            
                        mask_patches.append(mask)
                    except Exception as e:
                        print(f"Error processing patch: {e}")
                        # Create empty mask with the same shape as the input patch
                        mask_patches.append(np.zeros(patches[i+j].shape[:2], dtype=np.uint8))
            
            # Debug information
            print(f"Original image shape: {img.shape}")
            print(f"Number of patches: {len(patches)}")
            print(f"Patch shape: {patches[0].shape}")
            print(f"Mask patch shape: {mask_patches[0].shape}")
            print(f"Indices example: {indices[0]}")
            
            try:
                # Combine the patch predictions using EMPatches
                result_mask = self._merge_patches_emp(mask_patches, indices, mode='avg')
                results.append(result_mask)
            except ValueError as e:
                print(f"Error merging patches: {e}")
                # Fallback: create a mask the same size as the original image
                blank_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                results.append(blank_mask)
        
        return results

    def batch_inference_with_emp_patches(self, image_dir: str, file_extention: str, output_path: str, 
                                      device_id: int = 0, patch_size: int = 1360, overlap: float = 0.1):
        """
        Process a batch of images using EMPatches for patch extraction and merging.
        
        Args:
            image_path (str): directory path
            image_extension (str): extention
            output_path (str): output path
            device_id (int): GPU device ID
            patch_size (int): Size of each patch
            overlap (float): Overlap between patches (0-1)
        """
        start_time = time.time()
        assert type(device_id)==int, "{__file__} device_id should be integer"
        os.makedirs(output_path, exist_ok=True)
        image_list = self._data_set(image_dir, file_extention)
        
        for batch_name, batch_images in self._data_loader(image_list, self.batch_size*self.num_gpus):
            batch_preds = self.inference_with_emp_patches(batch_images, device_id, patch_size, overlap)
            # postprocess
            for img_name, pred in zip(batch_name, batch_preds):
                if pred is None:
                    print(f"Skipping image {img_name} as prediction is None.")
                    continue  

                output_file_path = os.path.join(output_path, img_name)
                pred = np.array(pred)

                if pred.dtype != np.uint8:
                    pred = (pred * 255).astype(np.uint8)

                cv2.imwrite(output_file_path, pred)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Batch inference with EMPatches took {execution_time:.4f} seconds")

if __name__ == '__main__': 
    # Reading the YAML file and getting parameters
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    params = yaml.safe_load(open(os.path.join(ROOT_DIR, "params/params_inference.yaml")))["segmentation_fingerprint"]
    args = Namespace(**params)
    
    # Get patch size and overlap from args, with defaults
    patch_size = getattr(args, 'patch_size', 1360)
    overlap = getattr(args, 'overlap', 0.1)
    use_empatches = getattr(args, 'use_empatches', True)

    # Initializing SegmentationInference
    inference_segmentation = SegmentationInference(checkpoint_path=args.checkpoint_path, batch_size=args.batch_size)

    # Choose between regular inference and EMPatches-based inference
    if use_empatches:
        results = inference_segmentation.batch_inference_with_emp_patches(
            image_dir=args.input_dir, 
            file_extention=args.file_extention,
            output_path=args.output_path,
            patch_size=patch_size,
            overlap=overlap
        )
    else:
        results = inference_segmentation.batch_inference(
            image_dir=args.input_dir, 
            file_extention=args.file_extention,
            output_path=args.output_path,
            patch_size=patch_size,
            overlap=overlap
        )
