import os
import sys
import argparse
from pathlib import Path
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
# ROOT_DIR = Path(__file__).parents[4].as_posix()
# sys.path.append(ROOT_DIR)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import tqdm
import yaml
import glob
import shutil
import argparse
import pandas as pd


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.multiprocessing import Process

from scripts.models.utils.EMA import EMA
from scripts.models.modules import TrainBaseModule
from scripts.models.utils import utils_train
from scripts.models.utils import  config 
from scripts.models.utils.convert_to_patches import CreateDataset
from scripts.models.utils.loss import connect_loss, convert_to_one_channel
from scripts.models.backbones.ncsnpp_generator_adagn import NCSNpp
from scripts.models.datareader import DataLoaderTrain, DataLoaderTest
from scripts.models.backbones.discriminator import Discriminator_large
from scripts.models.utils.posterior_coefficients import Posterior_Coefficients,Diffusion_Coefficients
from scripts.models.utils.Patching_Dataset import Patching_Dataset


conf = config.ConfigTrain()

class EnhancerTrain(TrainBaseModule):
    
    def __init__(self,
                 input_dir:str, 
                 image_extentions:str,
                 mask_extentions: str,
                 mpoint_extentions: str,
                 output_dir:str,
                 image_size: int, 
                 batch_size:int,
                 num_epochs:int,
                 save_ckpt_every:int, 
                 crop_input:bool,     
                 background_color_of_mask:int,          
                 num_proc_node:int=1, 
                 num_process_per_node:int=1,                  
                 full_image_train:bool=False, 
                 use_mpoints:bool = True,
                 padding_color:int=255,
                 fingerprint_type:str='latent',):
        
        self.image_size             = image_size
        self.input_dir              = input_dir
        self.image_extention        = image_extentions
        self.mask_extention         = mask_extentions
        self.mpoint_extention       = mpoint_extentions
        self.output_dir             = output_dir
        self.output_dir             = self.output_dir
        self.batch_size             = batch_size
        self.num_epochs             = num_epochs
        self.save_ckpt_every        = save_ckpt_every
        self.num_proc_node          = num_proc_node
        self.num_process_per_node   = num_process_per_node      
        self.full_image_train       = full_image_train
        self.crop_input             = crop_input                
        self.__set_parameters()
        self.config                 = conf.config 
        self.config.image_size      = self.image_size
        self.world_size             = self.num_proc_node * self.num_process_per_node
        self.use_mpoints            = use_mpoints
        self.background_color_of_mask = background_color_of_mask
        self.padding_color          = padding_color

        

        #Check if patch mode training is ON -> Apply Patching Method, otherwise, save without patching
        images_path = pd.read_csv(self.input_dir)["original_name"].tolist() if self.input_dir.endswith(".csv") else glob.glob(self.input_dir + f"/*{self.image_extention}")
        assert len(images_path) > 0, f"No image file found in directory {self.input_dir} with extention {self.image_extention}, please check image file extention and folder directory in params/params_train.yaml"
        print(f"{len(images_path)} images are found in {self.input_dir}")

        if fingerprint_type not in ["latent", "visible"]:
            raise ValueError(f"Fingerprint type should be 'latent' or 'visible', but got {fingerprint_type}.")
        self.fingerprint_type = fingerprint_type

        
        if os.path.isdir("enahancer_training_data_temp"):
            shutil.rmtree("enahancer_training_data_temp")
        

        orientation_extractor = None
        #Create Dataset 
        create_dataset = CreateDataset(images_path, orientation_extractor, 
                                       use_mpoints = self.use_mpoints, image_size = self.image_size,
                                       image_extention = self.image_extention,mpoint_extention = self.mpoint_extention, 
                                       mask_extention = self.mask_extention, bg_mask_color = self.background_color_of_mask)
        create_dataset.save_patches(is_csv = self.input_dir.endswith(".csv")) if not self.full_image_train else create_dataset.save_full_image(is_csv = self.input_dir.endswith(".csv"))
        
        
        
    def __set_parameters(self):
        self.fm_transform = transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize((self.image_size,self.image_size))]) 
                         
        self.train_transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),                
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])     
                                                                                  
                                  
    def __train_val_test_dataloader(self, rank):         
                          
        dataset = DataLoaderTrain(root = "enahancer_training_data_temp/images", image_extentions=".png", image_size = self.image_size, 
                                  transform=self.train_transform, transform_mask = self.fm_transform, crop_image = self.crop_input, use_minutiae = self.use_mpoints)
        dataset_val = DataLoaderTest(root  = "enahancer_training_data_temp/images", image_extentions=".png", image_size = self.image_size,
                                      transform=self.train_transform, transform_mask = self.fm_transform, crop_image = self.crop_input, use_minutiae = self.use_mpoints)
        
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=self.world_size,rank=rank)
        self.data_loader = torch.utils.data.DataLoader(dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=True,
                                                       sampler=self.train_sampler,
                                                       drop_last = True)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val,num_replicas=self.world_size,rank=rank)
        self.data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                           batch_size=self.batch_size,
                                                           shuffle=False,
                                                           num_workers=0,
                                                           pin_memory=True,
                                                           sampler=self.val_sampler,
                                                           drop_last = True)     
        
        
        print("Datasets have been loaded")
        print(f"Number of batches for training data    : {len(self.data_loader)}")
        print(f"Number of batches for validation data  : {len(self.data_loader_val)}")
        print("*" * 50)

        
                                                                     
    def __train_one_step(self, epoch, iteration, image, mask): #mpoint_mask
        mask        = (mask > 0) * 1.0
        # if self.use_mpoints:
        #     mpoint_mask = (mpoint_mask > 0) * 1.0
        # orientation = (orientation > 0) * 1.0
        
        #Train With Real 
        for p in self.disc_diffusive_1.parameters():  
            p.requires_grad = True       
                        
        self.disc_diffusive_1.zero_grad()
        # @Borhan: real_data1 => real_image && real_data2 => real_mask
        real_image        = image.to(self.DEVICE, non_blocking=True)
        real_mask         = mask.to(self.DEVICE, non_blocking=True)
        # if self.use_mpoints:
        #     real_mpoint_mask  = mpoint_mask.to(self.DEVICE, non_blocking=True)
        # real_orientation  = orientation.to(self.DEVICE, non_blocking=True)   
        
        # @Borhan: t1 and t2 must be reversed if you want to follow the syndiff strictly
        # @Borhan: (real_image, t2)    &&     (real_mask, t1)
        t1 = torch.randint(0, self.config.num_timesteps, (real_mask.size(0),), device=self.DEVICE)
        t2 = torch.randint(0, self.config.num_timesteps, (real_image.size(0),), device=self.DEVICE)
        
        rm1_t, rm1_tp1   = utils_train.q_sample_pairs(self.coeff, real_mask, t1)
        rm1_t.requires_grad = True
        
        x2_t, x2_tp1 = utils_train.q_sample_pairs(self.coeff, real_image, t2)     
        x2_t.requires_grad = True
        # rm1_t_with_ori   = torch.cat((rm1_t, real_orientation), axis = 1)
        # rm1_tp1_with_ori = torch.cat((rm1_tp1, real_orientation), axis = 1)
        # rm1_t_with_ori.requires_grad = True
        print(f"rm1_t shape: {rm1_t.shape}, rm1_tp1 shape: {rm1_tp1.shape}")
        print(f"x2_t shape: {x2_t.shape}, x2_tp1 shape: {x2_tp1.shape}")    
        D1_real = self.disc_diffusive_1(rm1_t, t1, rm1_tp1.detach()).view(-1)
        # D2_real = self.disc_diffusive_1(x2_t, t2, x2_tp1.detach()).view(-1)
    
        errD1_real = F.softplus(-D1_real)
        errD1_real = errD1_real.mean()            
        
        # errD2_real = F.softplus(-D2_real)
        # errD2_real = errD2_real.mean()
        
        errD_real = errD1_real# + errD2_real # 
        # errD_real = errD1_real + errD2_real
        errD_real.backward(retain_graph=True)             
        
        # @Borhan x1_t ===  === rm1_t
        if self.global_step % self.config.lazy_reg == 0:
            grad1_real = torch.autograd.grad(
                    outputs=D1_real.sum(), inputs=rm1_t, create_graph=True, 
                    )[0]
            grad1_penalty = (
                        grad1_real.view(grad1_real.size(0), -1).norm(2, dim=1) ** 2
                        ).mean()            
        
            grad_penalty = self.config.r1_gamma / 2 * grad1_penalty #args.r1_gamma / 2 * grad1_penalty + 
            grad_penalty.backward()            
        
        
        #Train With Fake
        latent_z1 = torch.randn(self.batch_size, self.config.nz, device=self.DEVICE)
        # latent_z2 = torch.randn(self.batch_size, self.config.nz, device=self.DEVICE)
        
        # @Borhan: Convert_to_one_channel is a custom function
        rm1_0_predict_diff,  rm1_0_predict_diff_aux = self.gen_diffusive_1(torch.cat((rm1_tp1.detach(),real_image),axis=1), t1, latent_z1) 
        rm1_0_predict_diff = convert_to_one_channel(rm1_0_predict_diff, self.batch_size, self.hori_translation, self.verti_translation, self.image_size)
        
        # x2_0_predict = self.gen_non_diffusive_1to2(real_data1)
        
        # @Borhan FIXME: 
        x1_pos_sample = utils_train.sample_posterior(self.pos_coeff, rm1_0_predict_diff, rm1_tp1, t1)
        # x1_pos_sample = torch.cat((x1_pos_sample, real_orientation), axis = 1)
        output1       = self.disc_diffusive_1(x1_pos_sample, t1, rm1_tp1.detach()).view(-1)
                    
        

        errD1_fake = F.softplus(output1)
        errD_fake = errD1_fake.mean() #+ errD2_fake.mean() # 
        errD_fake.backward()                    
        errD = errD_real + errD_fake

        self.optimizer_disc_diffusive_1.step()   
                
        
        #G part
        for p in self.disc_diffusive_1.parameters():
            p.requires_grad = False
            

        self.gen_diffusive_1.zero_grad()     
                    
        t1 = torch.randint(0, self.config.num_timesteps, (real_mask.size(0),), device=self.DEVICE)
        
        rm1_t, rm1_tp1 = utils_train.q_sample_pairs(self.coeff, real_mask, t1) 
        
                        
        latent_z1 = torch.randn(self.batch_size, self.config.nz ,device=self.DEVICE)   
        
        

        rm1_0_predict_diff, rm1_0_predict_diff_aux = self.gen_diffusive_1(torch.cat((rm1_tp1.detach(),real_image),axis=1), t1, latent_z1) 
        rm1_0_predict_diff_mask = convert_to_one_channel(rm1_0_predict_diff, self.batch_size, self.hori_translation, self.verti_translation, self.image_size)
        
        x1_pos_sample = utils_train.sample_posterior(self.pos_coeff, rm1_0_predict_diff_mask, rm1_tp1, t1)
        # x1_pos_sample = torch.cat((x1_pos_sample, real_orientation), axis = 1)
        output1 = self.disc_diffusive_1(x1_pos_sample, t1, rm1_tp1.detach()).view(-1)
        
        errG1 = F.softplus(-output1)
        errG1 = errG1.mean()
        
        errG_adv = errG1# + errG2 #      
        
        
        errG1_seg     = self.bicon_loss(rm1_0_predict_diff, real_mask.float())
        errG1_seg_aux = self.bicon_loss(rm1_0_predict_diff_aux, real_mask.float())

        #Add Minutiae Loss
        # if self.use_mpoints:
        #     errG1_seg_minutiae = self.segmentation_loss((rm1_0_predict_diff_mask * real_mpoint_mask), (real_mask * real_mpoint_mask))
            
        errG_L1 =  errG1_seg + 0.3 * errG1_seg_aux if self.use_mpoints else errG1_seg + 0.3 * errG1_seg_aux                                          
        torch.autograd.set_detect_anomaly(True)
        
        errG = 1 * errG_L1 + errG_adv
        errG.backward()
        
        self.optimizer_gen_diffusive_1.step()
        self.global_step += 1

        # self.errG1_seg_minutiae = errG1_seg_minutiae if self.use_mpoints else "N/A"
        self.errG_L1            = errG_L1
        self.errG_adv           = errG_adv
        self.errG               = errG
        self.errD               = errD
        # if self.use_mpoints:
        #     print('epoch {} iteration{}, G1-Minutiae : {}, G-L1: {}, G-Adv : {},  G-Sum: {}, D Loss: {}'.format(epoch,iteration, errG1_seg_minutiae.item(), errG_L1.item(), errG_adv.item(), errG.item(), errD.item()))
        # else:
        #     print('epoch {} iteration{}, G-L1: {}, G-Adv : {},  G-Sum: {}, D Loss: {}'.format(epoch,iteration, errG_L1.item(), errG_adv.item(), errG.item(), errD.item()))
        if iteration % 10 == 0:  
            rm1_t = torch.cat((torch.randn_like(real_mask),real_image),axis=1)
            fake_masks = self.sample_from_model(rm1_t)   
                                
            # Normalize images except the binary mask
            real_image_norm = (real_image - real_image.min()) / (real_image.max() - real_image.min())
            real_mask_norm = (real_mask - real_mask.min()) / (real_mask.max() - real_mask.min())
            # real_mpoint_mask_norm = (real_mpoint_mask - real_mpoint_mask.min()) / (real_mpoint_mask.max() - real_mpoint_mask.min())
            # real_orientation_norm = (real_orientation - real_orientation.min()) / (real_orientation.max() - real_orientation.min())

            # Ensure fake_masks is a float tensor with values in [0, 1]
            # fake_masks = fake_masks.float()
            fake_masks = fake_masks * 255.0
            # Concatenate all the images along the specified axis
            fake_sample2 = torch.cat((
                real_image_norm,
                real_mask_norm,
                # real_mpoint_mask_norm,
                # real_orientation_norm,
                fake_masks
            ), axis=-1)

            # Save the concatenated image without further normalization
            torchvision.utils.save_image(fake_sample2, os.path.join(self.output_dir, 'training_smaple_inference_epoch_{}.png'.format(epoch)), normalize=False)
                                        
    def __train_one_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        with tqdm.tqdm(total=len(self.data_loader), desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="Batch") as pbar:
            if self.use_mpoints:
                for iteration, (image, mask) in enumerate(self.data_loader): #mpoint_mask
                    self.__train_one_step(epoch, iteration, image, mask) #mpoint_mask
                    pbar.set_postfix({
                        'G1-Minutiae': f'{self.errG1_seg_minutiae.item():.4f}',
                        'G-L1': f'{self.errG_L1.item():.4f}',
                        'G-Adv': f'{self.errG_adv.item():.4f}',
                        'G-Sum': f'{self.errG.item():.4f}',
                        'D Loss': f'{self.errD.item():.4f}'
                    })
                    pbar.update(1)                    
                    
            else:
                for iteration, (image, mask) in enumerate(self.data_loader): #mpoint_mask
                    self.__train_one_step(epoch, iteration, image, mask) #mpoint_mask
                    pbar.set_postfix({
                        'G-L1': f'{self.errG_L1.item():.4f}',
                        'G-Adv': f'{self.errG_adv.item():.4f}',
                        'G-Sum': f'{self.errG.item():.4f}',
                        'D Loss': f'{self.errD.item():.4f}'
                    })
                    pbar.update(1)                    
                        
        if not self.config.no_lr_decay:            
            self.scheduler_gen_diffusive_1.step()            
            self.scheduler_disc_diffusive_1.step()               
            
     
        if epoch % self.save_ckpt_every == 0:
            # Save generator state with additional parameters
            checkpoint_gen = { 
                'full_image': self.full_image_train, 
                'padding_color': self.padding_color,  
                'image_size': self.image_size,
                'state_dict': self.gen_diffusive_1.state_dict(),
                'fingerprint_type': self.fingerprint_type
            }

            torch.save(checkpoint_gen, os.path.join(self.output_dir, 'gen_diffusive_1_{}.pth'.format(epoch)))

            torch.save(self.disc_diffusive_1.state_dict(), os.path.join(self.output_dir, 'disc_diffusive_1_{}.pth'.format(epoch)))            
    
    def train(self, rank, gpu):
        
        torch.manual_seed(self.config.seed + rank)
        torch.cuda.manual_seed(self.config.seed + rank)
        torch.cuda.manual_seed_all(self.config.seed + rank)    
        self.DEVICE = torch.device('cuda:{}'.format(gpu))    
        self.setup(rank, gpu)
        
        self.global_step, epoch, init_epoch = 0, 0, 0    
        for epoch in range(init_epoch, self.num_epochs): 
            self.__train_one_epoch(epoch)                                                                   
                
    def setup(self, rank, gpu):
                
        self.segmentation_loss = nn.BCEWithLogitsLoss()
        self.segmentation_loss.to(self.DEVICE)
                
        self.__train_val_test_dataloader(rank)                
        self.to_range_0_1 = lambda x: (x + 1.) / 2.       
            
        self.hori_translation = torch.zeros([1, 1, self.image_size, self.image_size])
        for i in range(self.image_size-1):
            self.hori_translation[:,:,i,i+1] = torch.tensor(1.0)
        self.verti_translation = torch.zeros([1, 1, self.image_size, self.image_size])
        for j in range(self.image_size-1):
            self.verti_translation[:,:,j,j+1] = torch.tensor(1.0)
            
        self.hori_translation  = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()    
        
        self.bicon_loss = connect_loss(self.hori_translation, self.verti_translation)
        self.bicon_loss.to(self.DEVICE)          
        
        self.gen_diffusive_1 = NCSNpp(self.config, seg = True, is_three_inputs = False).to(self.DEVICE)
        self.config.num_channels=1
        self.disc_diffusive_1 = Discriminator_large(nc = 2, ngf = self.config.ngf, 
                                    t_emb_dim = self.config.t_emb_dim,
                                    act=nn.LeakyReLU(0.2)).to(self.DEVICE)
        
        utils_train.broadcast_params(self.gen_diffusive_1.parameters())
        utils_train.broadcast_params(self.disc_diffusive_1.parameters())
        self.optimizer_disc_diffusive_1 = optim.Adam(self.disc_diffusive_1.parameters(), lr=self.config.lr_d, betas = (self.config.beta1, self.config.beta2))
        self.optimizer_gen_diffusive_1  = optim.Adam(self.gen_diffusive_1.parameters() , lr=self.config.lr_g, betas = (self.config.beta1, self.config.beta2))
        if self.config.use_ema:
            self.optimizer_gen_diffusive_1 = EMA(self.optimizer_gen_diffusive_1, ema_decay=self.config.ema_decay)
            
        self.scheduler_gen_diffusive_1  = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_gen_diffusive_1 , self.num_epochs, eta_min=1e-5)
        self.scheduler_disc_diffusive_1 = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_disc_diffusive_1, self.num_epochs, eta_min=1e-5)

        
        #ddp
        self.gen_diffusive_1  = nn.parallel.DistributedDataParallel(self.gen_diffusive_1, device_ids=[gpu])
        self.disc_diffusive_1 = nn.parallel.DistributedDataParallel(self.disc_diffusive_1, device_ids=[gpu])
        
        exp_path = self.output_dir
        if rank == 0:
            if not os.path.exists(exp_path):
                os.makedirs(exp_path)
                # utils_train.copy_source(__file__, exp_path)
                # shutil.copytree('./backbones', os.path.join(exp_path, 'backbones'))
        
        
        self.coeff     = Diffusion_Coefficients(self.config, self.DEVICE)
        self.pos_coeff = Posterior_Coefficients(self.config, self.DEVICE)
        self.T         = utils_train.get_time_schedule(self.config, self.DEVICE) 
        print("Model Setup")              
        print(50 * "*")          
      
    def sample_from_model(self, x_init):
        x = x_init[:,[0],:] #Noise
        source = x_init[:,[1],:] #real_data_1
        with torch.no_grad():
            for i in reversed(range(self.config.num_timesteps)):
                t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)          
                t_time = t
                latent_z = torch.randn(x.size(0), self.config.nz, device=x.device)#.to(x.device)
                x_0, _ = self.gen_diffusive_1(torch.cat((x,source),axis=1), t_time, latent_z)
                x_0 = convert_to_one_channel(x_0, self.batch_size, self.hori_translation, self.verti_translation, self.image_size)
                x_new = utils_train.sample_posterior(self.pos_coeff, x_0[:,[0],:], x, t)
                x = x_new.detach()           
        return x      
                    
    def init_processes(self, rank, size, fn, args):
        os.environ['MASTER_ADDR'] = args.master_address
        os.environ['MASTER_PORT'] = args.port_num
        torch.cuda.set_device(args.local_rank)
        gpu = args.local_rank
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)    
        fn(rank, gpu)
        dist.barrier()
        self.cleanup()  

    def cleanup(self):
        dist.destroy_process_group()   
                            

#%%
if __name__ == '__main__':    
    params = yaml.safe_load(open('params/diff_gan_train.yaml'))["diffgan"]
    size = params["num_process_per_node"]
    ehancer_train = EnhancerTrain(image_size       = params["image_size"]               ,input_dir            = params["input_dir"],
                                  output_dir       = params["output_dir"]               ,mask_extentions      = params["mask_extentions"],
                                  batch_size       = params["batch_size"]               ,num_epochs           = params["num_epochs"],
                                  save_ckpt_every  = params["save_ckpt_every"]          ,num_proc_node        = params["num_proc_node"],
                                  num_process_per_node = params["num_process_per_node"] ,full_image_train     = params["full_image_train"],
                                  image_extentions = params["image_extentions"]         ,crop_input           = params["crop_input"],
                                  use_mpoints      = params["use_mpoints"]              ,fingerprint_type     = params["fingerprint_type"],
                                  mpoint_extentions = params["mpoint_extentions"]    ,background_color_of_mask = params["background_color_of_mask"])
                                  
    
    if size > 1:
        processes = []
        for rank in range(size):
            ehancer_train.config.local_rank = rank
            global_rank = rank + ehancer_train.config.node_rank * ehancer_train.num_process_per_node
            global_size = ehancer_train.num_proc_node * ehancer_train.num_process_per_node
            ehancer_train.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (ehancer_train.config.node_rank, rank, global_rank))
            p = Process(target=ehancer_train.init_processes, args=(global_rank, global_size, ehancer_train.train, ehancer_train.config))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()

    else:
        ehancer_train.init_processes(0, size, ehancer_train.train, ehancer_train.config)

