import os
import sys
import argparse
from pathlib import Path
from typing import Union, List, Tuple
ROOT_DIR = Path(__file__).parents[2].as_posix()
sys.path.append(ROOT_DIR)
import cv2
import yaml
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from scripts.models.utils import utils, config
from scripts.models.modules import InferenceBaseModule
from scripts.models.utils.loss import convert_to_one_channel
from scripts.models.utils.posterior_coefficients import Posterior_Coefficients
from scripts.models.utils.Patching_Dataset import Patching_Dataset
from scripts.models.backbones.ncsnpp_generator_adagn import NCSNpp


conf = config.ConfigInference()
# export NCCL_SOCKET_IFNAME=eth0 # for multi-gpu on cluster first run this command
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
class InferenceEnhancer(InferenceBaseModule):

    def __init__(self, checkpoint_path: str, step: int, batch_size:int=1):
        self.step = step
        assert os.path.isfile(checkpoint_path), "Please Check Your Checkpoint Path"
        checkpoint_file = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.image_size = checkpoint_file["image_size"]
        self.padding_color = checkpoint_file["padding_color"]
        self.full_image = checkpoint_file["full_image"]
        self.state_dict = checkpoint_file["state_dict"]
        self.config = conf.config
        self.config.image_size = self.image_size
        
        self.__set_parameters()
        self.generator = self._load_model().eval()
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            device_ids = list(range(self.num_gpus))
            self.generator = nn.DataParallel(self.generator, device_ids=device_ids)
        # self.T = utils.get_time_schedule(self.config)
        self.POS_COEF = Posterior_Coefficients(self.config)
        self.batch_size = self.set_batch_size(batch_size)
    
    def to_range_0_1(self, x):
        return (x + 1.) / 2.
    
    def __set_parameters(self):
    
        if self.full_image:
            self.transforms = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),                
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])
        else: 
            self.transforms = transforms.Compose([              
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5))
                ])     
        self.to_pil_image = transforms.ToPILImage()  

    def set_batch_size(self, batch_size: int) -> int:
        """
        Set the effective batch size based on the number of GPUs and the input batch size.
        
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")
        
        # self.effective_batch_size = batch_size * self.num_gpus
        self.effective_batch_size = batch_size 
        self.config.effective_batch_size = self.effective_batch_size
        return self.effective_batch_size
                      
    def _load_model(self):
        for key in list(self.state_dict.keys()):
            self.state_dict[key[7:]] = self.state_dict.pop(key)            
        gen_diffusive_1 = NCSNpp(self.config, seg = True)
        gen_diffusive_1.load_state_dict(self.state_dict)
        return gen_diffusive_1
    
    def _sample_from_model(self, x_init:torch.Tensor, device="cuda:0")->torch.Tensor:
        """
        get model predict
        x_init (torch.Tensor): input tensor 3D [b, w, h]
        Return: torch.Tensor
        """
        x = x_init[:, [0], :].to(device)  # Noise
        source = x_init[:, [1], :].to(device)  # real_data_1
        
        hori_translation = torch.zeros([1, 1, self.image_size, self.image_size])
        for i in range(self.image_size - 1):
            hori_translation[:, :, i, i + 1] = torch.tensor(1.0)

        verti_translation = torch.zeros([1, 1, self.image_size, self.image_size])
        for j in range(self.image_size - 1):
            verti_translation[:, :, j, j + 1] = torch.tensor(1.0)

        hori_translation = hori_translation.float().to(device)
        verti_translation = verti_translation.float().to(device)        
        with torch.no_grad():
            for i in reversed(range(self.config.num_timesteps)):
                
                # t_time = torch.full((x.size(0),), i, dtype=torch.int64).to(device)
                # latent_z = torch.randn(x.size(0), self.config.nz, device=device)  

                t_time = torch.full((1,), i, dtype=torch.int64).to(device).repeat(x.size(0))
                latent_z = torch.randn(1, self.config.nz, device=device).repeat(x.size(0),1)
                inp = torch.cat((x, source), axis=1).to(device)
                
                x_0, _ = self.generator(inp, t_time, latent_z)
                x_0 = convert_to_one_channel(x_0, x_0.shape[0], hori_translation, verti_translation, self.image_size)

                x_new = utils.sample_posterior(self.POS_COEF, x_0[:, [0], :], x, t_time, device)
                x = x_new.detach()
        return x

    def _preprocess_pred_patch(self, patch: torch.Tensor)->torch.Tensor:
        """
        prepaire patch to apply to model
        patch (torch.Tensor): input pathces tesnor [b, w, h]
        
        Return :
            batch input (torch.Tensor): [b, 2, w, h] apply a random noise to the input patches
        """
        patchs = patch.cpu().detach().numpy()
        results = []
        for p in patchs:
            # patch = Image.fromarray(np.uint8(p))
            # results.append(self.transforms(patch))
            results.append(self.transforms(p))
        
        batchs_patches = torch.from_numpy(np.array(results))
        z = torch.randn((1,self.image_size, self.image_size), device=batchs_patches.device).repeat(batchs_patches.shape[0],1,1,1)
        batch_input = torch.cat((z, batchs_patches), axis=1)
        return batch_input

    def _preprocess_patch(self, images:Union[List[np.array], np.array])->List[np.array]:
        """
        check if input image is Image.Image or list of Image.Image. then return a list of Image.Image
        Return :
            image (List[Image.Image])
        """
        if type(images) == np.ndarray:
            assert len(images.shape) == 2, f"input image should be GRAY"
            return [images]
        elif type(images) == list:
            batch_image = []
            for img in images:
                assert len(img.shape) == 2, f"input image should be GRAY"
                batch_image.append(img)
            return batch_image
        else:
            raise  f"input image should be list or PIL.Image but is {type(images)}"
    
    def _postprocess_patch(self, mask:np.array, latent_en:np.array, pad_left:int, pad_top:int, original_w:Tuple[int, int], original_h:Tuple[int, int])-> Image.Image:
        """
        get enhancer output then resize it to input original image. and convert to binary image.
        """
        mask[mask==0] = 1
        pred = latent_en / mask
        # pred = Image.fromarray(pred)
        pred = utils.remove_padding_cv2(pred, pad_left, pad_top, (original_w, original_h))
        pred = (np.array(pred) * 255).astype(np.uint8)
        return pred
    
    def _post_model(self, pred:torch.tensor, latent_en:np.array, mask:np.array, patch_ind:Tuple[int, int])-> Tuple[np.array, np.array]:
        """
        assign predicted model patch to input variables (latent_en, mask) with respect of the index of the patch.
        
        pred (torch.tensor): patch predicted
        latent_en (np.array): enhance variable
        mask (np.array): binary enhance variable
        patch_ind (Tuple[int, int]): correspond patch to the variables [latent_en, mask]
        
        Return:
            latent_en: apply new predict slice
            mask: apply new binary predict slice
        """
        pred = self.to_range_0_1(pred) ; pred = pred/pred.max()
        row_ind, col_ind = patch_ind
        latent_en[(row_ind - self.image_size):row_ind, (col_ind - self.image_size):col_ind] += pred.cpu().detach().numpy().squeeze(0)
        mask[(row_ind - self.image_size):row_ind, (col_ind - self.image_size):col_ind] += 1
        return latent_en, mask

    def _adjust_padding(self, image:Image.Image)-> Tuple[np.array, int, int]:
        """
        apply padding to input image
        """
        pad_left, pad_top, image = utils.pad_to_min_size_cv2(image, self.image_size, self.padding_color)
        return np.array(image), pad_left, pad_top
    
    def _prepare_dataset_patching(self, images: Union[List[Image.Image], Image.Image])->Tuple[Patching_Dataset, List[int], List[int]]:
        """
        get list of images. make a dataset containe patches of the images
        image (Image.Image or list of it)
        Returns:
            dataset (torch.Dataset): contain all patches of the image or images
            image_info (Tuple): a list Tuple, each list element contain image number and len of patches of that image
            images_shape (Tuple):  a list Tuple, each  list element contain (original image shape, adjusted image shape, left padded number pixel, top padded number pixel)
        """
        image_info = []
        images_shape = []
        for img_no, image in enumerate(self._preprocess_patch(images)):
            original_size = image.shape[:2][::-1]
            # padding image to pretrained weight image size
            image, pad_left, pad_top = self._adjust_padding(image)
            images_shape.append((original_size, image.shape, pad_left, pad_top))
            #Patching procedure
            if img_no == 0:
                dataset = Patching_Dataset(image, self.image_size,self.image_size, self.step)
                image_info.append((img_no, dataset.__len__()))
            else:
                tmp = Patching_Dataset(image, self.image_size,self.image_size, self.step)
                dataset += tmp
                image_info.append((img_no, tmp.__len__()))
        return dataset, image_info, images_shape
    
    def _process_patching(self, dataset:Patching_Dataset, batch_size:int, device:str)->Tuple[List[torch.Tensor], List[int]]:
        """
        apply to DataLoader input patch dataset and apply to model each patch 
        
        dataset (Patching_Dataset): input patch dataset
        batch_size (int):
        device (str): string device to apply process

        Return:
            preds (torch.Tensor): output model list
            patch_inds (list): predicted model patch index
        """
        preds = []
        patch_inds = []
        patches = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
        for step, (patch_ind, patch) in enumerate(patches):
            batch_image = self._preprocess_pred_patch(patch).to(device)
            preds.append(self._sample_from_model(batch_image, device))
            patch_inds.append(patch_ind)
        return preds, patch_inds
    
    def _convert_patcher_to_image(self, preds:List[torch.Tensor], patch_inds:List[int], image_info: List[int], images_shape: List[int])->List[Image.Image]:
        """
        get all predicted patches then extract it and convert to output image.

        preds (List[torch.Tensor]): list of output model.
        patch_inds (List[int]): list of index of the output model patch
        image_info (List[int]):  details or map list of the patch output model with respect of the original image
        images_shape (List[int]): shape of original image, shape of padded image, number or the left pad pixel and number of the top pad pixel

        Return:
            enhanced_images (List[Image.Image): list of the predicted enhanced fingerprint.

        """
        # postprocess
        patch_inds = torch.concat(patch_inds, axis=0)
        preds = torch.concat(preds, axis=0)
        enhanced_images = []
        for img_info, img_shape in zip(image_info, images_shape):
            try:
                # split preds base on image_info
                img_no, num_patches = img_info
                if img_no==0:
                    start_idx = 0
                    end_idx = num_patches
                else:
                    start_idx = end_idx
                    end_idx += num_patches
                selected_preds = preds[start_idx:end_idx]
                selected_inds = patch_inds[start_idx:end_idx]
                # initializ latent and mask variables
                original_size, padded_shape, pad_left, pad_top = img_shape
                latent_en = np.zeros(padded_shape)
                mask = np.zeros(padded_shape)
                for patch_ind, pred in zip(selected_inds, selected_preds):
                    latent_en, mask = self._post_model(pred, latent_en, mask, patch_ind)
                enhanced_images.append(self._postprocess_patch(mask, latent_en, pad_left, pad_top, *original_size))
            except:
                enhanced_images.append(None)
                continue

        return enhanced_images

    def _inference_patching(self, images: Union[List[Image.Image],Image.Image], batch_size:int, device:str)-> List[Image.Image]:
        """
        inference model in patching mode.
        images (Union[List[Image.Image],Image.Image]): input image or images
        batch_size (int):
        device (str): device 'cuda:x'
        
        Return:
            list of the Enahnced fingeprint
        """
        dataset, image_info, images_shape = self._prepare_dataset_patching(images)
        preds, patch_inds = self._process_patching(dataset, batch_size, device)
        enhanced_images = self._convert_patcher_to_image(preds, patch_inds, image_info, images_shape)
        return enhanced_images
    
    def _inference_full(self, image: Union[List[Image.Image],Image.Image], device:str)-> List[Image.Image]:
        """
        inference model in full image mode.
        
        image (Union[List[Image.Image],Image.Image]): input image or images
        device (str): device 'cuda:x'
        
        Return:
            list of the Enahnced fingerprint
        """
        batch_image, batch_shape = self._preprocess_full(image,device)
        preds = self._sample_from_model(batch_image, device)
        results = []
        for pred, img_shape in zip(preds, batch_shape):
            try:
                results.append(self._postprocess_full(pred, *img_shape))
            except:
                results.append(None)
                continue
        return results
            
    def _postprocess_full(self, image, w:int, h:int)-> Image.Image:
        """
        get enhancer output then resize it to input original image. and convert to binary image.
        """
        image = self.to_range_0_1(image) ; image = image/image.max()
        output = image.cpu().detach().numpy().squeeze(0) 
        # output = Image.fromarray(output)
        # output = output.resize((w, h), resample=Image.NEAREST)
        # return (np.array(output) * 255).astype(np.uint8)
        output = (output * 255).astype(np.uint8)
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_NEAREST)
        return output

    def _preprocess_full(self, images:Image.Image, device:str)->Tuple[torch.Tensor, List[int]]:
        """
        convert input image to torch.Tensor
        images (Image.Image): input image

        """
        if type(images) == np.ndarray:
            assert len(images.shape) == 2, f"input image should be GRAY"
            images = Image.fromarray(images)
            full_image = self.transforms(images).unsqueeze(0)
            utils.set_seed(self.config.seed)
            z = torch.randn_like(full_image)
            x2_t = torch.cat((z, full_image), axis=1)
            return x2_t, [images.size]
        elif type(images) == list:
            batch_image = []
            batch_shape = []
            for img in images:
                assert len(img.shape) == 2, f"input image should be GRAY"
                img = Image.fromarray(img)
                full_image = self.transforms(img).to(device)
                utils.set_seed(self.config.seed)
                z = torch.randn_like(full_image)
                x2_t = torch.cat((z, full_image), axis=0)
                batch_shape.append(img.size)
                # batch_image.append(x2_t)
                batch_image.append(x2_t.cpu())
            # return torch.from_numpy(np.array(batch_image)), batch_shape
            return torch.from_numpy(np.array(batch_image)).to(device), batch_shape
        else:
            raise  f"input image should be list or PIL.Image but is {str(type(images))}"

    def inference(self, image:Union[List[Image.Image], Image.Image], device_id:int=0)->List[Image.Image]:
        """
        Performs inference on a single image using the trained model.

        Parameters:
        image (numpy.ndarray): The input image to be processed.
        batch_size (int):
        device_id (int): gpu number
        Returns:
        tuple: A tuple containing the original image and the processed image.
        """
        assert type(device_id)==int, "device_id should be integer"
        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed_all(self.config.seed)
        # utils.set_seed(self.config.seed)
        self.generator.to(device)
        if self.full_image:
            pred = self._inference_full(image, device)
        else:
            if self.batch_size is None:
                if type(image)==Image.Image:
                    self.batch_size = 1
                elif type(image)==list:
                    self.batch_size = len(image)
                else:
                    raise "input image should be list or PIL.IMAGE.IMAGE"
            assert self.batch_size>0, "in patching mode path size should be >0"
            pred = self._inference_patching(image, self.batch_size, device)
        return pred

    def _data_set(self, image_dir:str, file_extention:str):
        if os.path.isdir(image_dir):
            out_list = [p for p in Path(image_dir).glob(f"**/*{file_extention}")]
        elif os.path.isfile(image_dir):
            out_list = [Path(image_dir)]
        else:
            raise ValueError("image_path is not valid")
        assert len(out_list), "does not found any image in the directory"
        return out_list

    def _data_loader(self, ds:List, batch_size:int):
        batch_name = []
        batch_images = []
        for idx, p in enumerate(ds):
            image = cv2.imread(p.as_posix(), 0)
            # batch_name.append(p.name)
            name = p.name.split(".")[0]
            batch_name.append(name)
            batch_images.append(image)
            if len(batch_images) % batch_size==0 or idx==(len(ds)-1):
                yield batch_name, batch_images
                batch_name = []
                batch_images = []

    def batch_inference(self, image_dir:str, file_extension:str, output_path:str, device_id:int=0):
        """
        apply batch inference
        """
        assert type(device_id)==int, "device_id should be integer"
        os.makedirs(output_path, exist_ok=True)
        image_list = self._data_set(image_dir, file_extension)
        batch_size = self.batch_size * self.num_gpus
        for  batch_name, batch_images  in self._data_loader(image_list,  batch_size):
            batch_preds = self.inference(batch_images)
            # postprocess
            for img_name, pred in zip(batch_name, batch_preds):
                output_file_path = os.path.join(output_path, img_name+".png")
                _, img = cv2.threshold(pred, 200, 255, cv2.THRESH_BINARY)
                cv2.imwrite(output_file_path, img)


if __name__ == '__main__':
    import time
    params = yaml.safe_load(open(os.path.join(ROOT_DIR, "params/diff_gan_inference.yaml")))["diffgan"]
    args = argparse.Namespace(**params)
    t0 = time.time()
    enhancer_inference = InferenceEnhancer(args.checkpoint_path, args.step, args.batch_size)
    enhancer_inference.batch_inference(args.input_dir, args.file_extention, args.output_path, 0)
    print(f"Total Time: {time.time()-t0}")
  
  