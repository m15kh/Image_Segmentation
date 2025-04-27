import os
import sys
import argparse
from pathlib import Path
ROOT_DIR = Path(__file__).parents[2].as_posix()
sys.path.append(ROOT_DIR)
import yaml
from scripts.models.inference import InferenceEnhancer
import time


params = yaml.safe_load(open(os.path.join(ROOT_DIR, "params/inference.yaml")))["diffgan"]
args = argparse.Namespace(**params)
t0 = time.time()
enhancer_inference = InferenceEnhancer(args.checkpoint_path, args.step, args.batch_size)
enhancer_inference.batch_inference(args.input_dir, args.file_extention, args.output_path, 0)
print(f"Total Time: {time.time()-t0}")

