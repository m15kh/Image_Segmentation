import os
import sys
from pathlib import Path
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
# ROOT_DIR = Path(__file__).parents[4].as_posix()
# sys.path.append(ROOT_DIR)
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import yaml
from scripts.models.train import EnhancerTrain
from scripts.models.train import Process

if __name__ == '__main__':    
    params = yaml.safe_load(open('params/train.yaml'))["diffgan"]
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

