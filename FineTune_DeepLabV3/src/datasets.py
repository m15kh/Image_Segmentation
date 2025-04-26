import glob
import numpy as np
import torch
import albumentations as A

from utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2

def get_images(root_path):
    train_images = glob.glob(f"{root_path}/train_images/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/train_masks/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/valid_images/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/valid_masks/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def normalize():
    """
    Transform to normalize image.
    """
    transform = A.Compose([
        A.Normalize(
            mean=[0.45734706, 0.43338275, 0.40058118],
            std=[0.23965294, 0.23532275, 0.2398498],
            always_apply=True
        )
    ])
    return transform

def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ])
    return valid_image_transform

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        norm_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.norm_tfms = norm_tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        # Convert string names to class values for masks.
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[index]).convert('RGB'))

        mask = cv2.bitwise_not(mask) #betwise
        # Make any pixel value above 200 as 255 for waterbody.
        im = mask >= 200
        mask[im] = 255
        mask[np.logical_not(im)] = 0

        image = self.norm_tfms(image=image)['image']
        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Get colored label mask.
        mask = get_label_mask(mask, self.class_values, self.label_colors_list)
       
        image = np.transpose(image, (2, 0, 1))
        
        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.long) 

        return image, mask

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)
    norm_tfms = normalize()

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        norm_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        norm_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)

    return train_data_loader, valid_data_loader