import os
import os
import sys
from typing import Tuple, List
from pathlib import Path
ROOT_DIR = Path(__file__).parents[2].as_posix()
sys.path.append(ROOT_DIR)
from PIL import Image
import numpy as np
import pandas as pd
from argparse import Namespace
import cv2
from contextlib import contextmanager
import psutil
import multiprocessing as mp
from multiprocessing import Pool


class TrainBaseModule:
    
    def __train_val_test_dataloader(self):
        pass
    
    def __set_parameters(self):
        pass
    
    def __train_one_step(self):
        pass
    
    def __train_one_epoch(self):
        pass
    
    def train(self):
        pass
    
    def validation_evaluation(self):
        pass
    
    def test_evaluation(self):
        pass


class InferenceBaseModule:
    
    def __set_parameters(self):
        pass
    
    def inference(self, image:np.array)->np.array:
        pass
    
    def _data_set(self):
        pass
    
    def _data_loader(self):
        pass

    def batch_inference(self, input_dir:str, file_extention:str, output_path:str, device:str):
        pass


def logging(p:str, args:Namespace):
    with open(p, mode="a") as f:
        log = ""
        for k , v in args.__dict__.items():
            log += f"{k}: {v} \n"
        log += f"="*20
        log += f"\n"
        f.write(log)
