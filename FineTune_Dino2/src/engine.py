import torch
import numpy as np

from tqdm import tqdm
from utils import draw_translucent_seg_maps
from metrics import pix_acc

def train(
    model,
    train_dataset,
    train_dataloader,
    device,
    optimizer,
    criterion,
    classes_to_train
):
    print('Training')
    model.train()
    train_running_loss = 0.0
    train_running_correct, train_running_label = 0, 0
    # Calculate the number of batches.
    num_batches = int(len(train_dataset)/train_dataloader.batch_size)
    prog_bar = tqdm(train_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    counter = 0 # to keep track of batch counter
    num_classes = len(classes_to_train)

    for i, data in enumerate(prog_bar):
        counter += 1
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)['out']

        ##### BATCH-WISE LOSS #####
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        ###########################

        # For pixel accuracy.
        labeled, correct = pix_acc(target, outputs, num_classes)
        train_running_label += labeled.sum()
        train_running_correct += correct
        train_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled.sum())
        #############################

        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################

        prog_bar.set_description(desc=f"Loss: {loss.detach().cpu().numpy():.4f} | PixAcc: {train_running_pixacc.cpu().numpy()*100:.2f}")
        
    ##### PER EPOCH LOSS #####
    train_loss = train_running_loss / counter
    ##########################

    ##### PER EPOCH METRICS ######
    # Pixel accuracy
    pixel_acc = ((1.0 * train_running_correct) / (np.spacing(1) + train_running_label)) * 100
    ##############################
    return train_loss, pixel_acc

def validate(
    model,
    valid_dataset,
    valid_dataloader,
    device,
    criterion,
    classes_to_train,
    label_colors_list,
    epoch,
    all_classes,
    save_dir
):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct, valid_running_label = 0, 0
    # Calculate the number of batches.
    num_batches = int(len(valid_dataset)/valid_dataloader.batch_size)
    num_classes = len(classes_to_train)

    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1
            data, target = data[0].to(device), data[1].to(device)
            outputs = model(data)['out']
            
            # Save the validation segmentation maps every
            # last batch of each epoch
            if i == num_batches - 1:
                draw_translucent_seg_maps(
                    data, 
                    outputs, 
                    epoch, 
                    i, 
                    save_dir, 
                    label_colors_list,
                )

            ##### BATCH-WISE LOSS #####
            loss = criterion(outputs, target)
            valid_running_loss += loss.item()
            ###########################

            # For pixel accuracy.
            labeled, correct = pix_acc(target, outputs, num_classes)
            valid_running_label += labeled.sum()
            valid_running_correct += correct
            valid_running_pixacc = 1.0 * correct / (np.spacing(1) + labeled.sum())
            #############################

            prog_bar.set_description(desc=f"Loss: {loss.detach().cpu().numpy():.4f} | PixAcc: {valid_running_pixacc.cpu().numpy()*100:.2f}")
        
    ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################

    ##### PER EPOCH METRICS ######
    # Pixel accuracy.
    pixel_acc = ((1.0 * valid_running_correct) / (np.spacing(1) + valid_running_label)) * 100.
    ##############################
    return valid_loss, pixel_acc