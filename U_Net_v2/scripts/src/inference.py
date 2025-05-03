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
        self.width = checkpoint["width"]
        self.height = checkpoint["height"]
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict)
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

    def inference(self, image: Union[List[np.array], np.array], device_id: int = 0) -> list[np.array]:
        """input a list of RGB images

        Args:
            images (list[np.array]): list of images

        Returns:
            list[np.array]: list of masks
        """
        
        device = f"cuda:{device_id}"
        batch_image, batch_shape = self._preprocess(image)

        batch_image = batch_image.to(device)
        self.model.to(device)
        with torch.no_grad():
            with autocast():
                preds = self.model(batch_image)
                 
        results = []
        for pred, img_shape in zip(preds, batch_shape):
            try:
                results.append(self._postprocess(pred, *img_shape))
            except:
                results.append(None)
                continue
        
        return results

    def batch_inference(self, image_dir: str, file_extention: str, output_path: str, device_id:int=0):
        """
        input image directory image
        write output mask to look for fingerprint

        Args:
            image_path (str): directory path
            image_extension (str): extention
            output_path (str): output path
        """

        start_time = time.time()
        assert type(device_id)==int, "{__file__} device_id should be integer"
        os.makedirs(output_path, exist_ok=True)
        image_list = self._data_set(image_dir, file_extention)
        
        for batch_name, batch_images in self._data_loader(image_list,self.batch_size*self.num_gpus):
            batch_preds = self.inference(batch_images, device_id)
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
        

if __name__ == '__main__': 
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

    # Reading the YAML file and getting the batch_size parameter
    params = yaml.safe_load(open(os.path.join(ROOT_DIR, "params/params_inference.yaml")))["segmentation_fingerprint"]
    args = Namespace(**params)

    # Initializing SegmentationInference with checkpoint_path and batch_size
    inference_segmentation = SegmentationInference(checkpoint_path=args.checkpoint_path, batch_size=args.batch_size) 
    # img = cv2.imread("/home/mehdi/Downloads/sufs_test_result/sufs_input/222.png")
    # res = inference_segmentation.inference(Image.fromarray(img))

    results = inference_segmentation.batch_inference(image_dir=args.input_dir, 
                                                    file_extention=args.file_extention,
                                                    output_path=args.output_path)
    