import torch
import torch.nn as nn
import os
import yaml
import sys
import time
sys.path.append(".")
from datasets import get_images, get_dataset, get_data_loaders
from engine import train, validate
from model import prepare_model
from config import ALL_CLASSES, LABEL_COLORS_LIST
from utils import save_model, SaveBestModel, save_plots
from loss import DiceBCELoss

# Load configuration from YAML file
config_path = '/home/ubuntu2/m15kh/Image_Segmentation/FineTune_DeepLabV3/params/train.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Extract parameters from config
EPOCHS = config['epochs']
LEARNING_RATE = config['learning_rate']
BATCH_SIZE = config['batch_size']
DATA_ROOT_PATH = config['data_root_path']
OUTPUT_DIR = config['output_dir']
VALID_PREDS_DIR = config['valid_preds_dir']
HEIGHT_SIZE = config['height_size']
WIDTH_SIZE = config['width_size']

print(f"Configuration loaded from {config_path}")
print(f"Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}")
print(f"Data root path: {DATA_ROOT_PATH}")
print(f"Output directory: {OUTPUT_DIR}")

if __name__ == '__main__':
    start_time = time.time()  # Record the start time

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VALID_PREDS_DIR, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = prepare_model(num_classes=len(ALL_CLASSES)).to(device)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = DiceBCELoss()

    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path=DATA_ROOT_PATH
    )

    classes_to_train = ALL_CLASSES

    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        width_size= WIDTH_SIZE,
        height_size = HEIGHT_SIZE
        
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=BATCH_SIZE
    )

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    train_loss, train_pix_acc = [], []
    valid_loss, valid_pix_acc = [], []
    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc = train(
            model,
            train_dataset,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train
        )
        valid_epoch_loss, valid_epoch_pixacc = validate(
            model,
            valid_dataset,
            valid_dataloader,
            device,
            criterion,
            classes_to_train,
            LABEL_COLORS_LIST,
            epoch,
            ALL_CLASSES,
            save_dir=VALID_PREDS_DIR
        )
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc.cpu())
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc.cpu())

        if config.get('save_best_model', True):
            save_best_model(
                valid_epoch_loss, epoch, model, OUTPUT_DIR
            )

        print(f"Train Epoch Loss: {train_epoch_loss:.4f}, Train Epoch PixAcc: {train_epoch_pixacc:.4f}")
        print(f"Valid Epoch Loss: {valid_epoch_loss:.4f}, Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}")
        print('-' * 50)

    save_model(EPOCHS, model, optimizer, criterion, OUTPUT_DIR)
    # Save the loss and accuracy plots.
    save_plots(
        train_pix_acc, valid_pix_acc, train_loss, valid_loss, OUTPUT_DIR
    )
    print('TRAINING COMPLETE')

    end_time = time.time()  # Record the end time
    total_time = end_time - start_time  # Calculate the total runtime
    print(f"Total Training Time: {total_time:.2f} seconds")