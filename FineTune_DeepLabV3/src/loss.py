import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        
    def forward(self, inputs, targets, smooth=1):
        # Handle multi-class segmentation
        # inputs shape: [batch_size, num_classes, height, width]
        # targets shape: [batch_size, height, width] with class indices
        
        # Get dimensions
        batch_size = inputs.size(0)
        num_classes = inputs.size(1)
        
        # Apply softmax to get class probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot = targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Calculate Dice coefficient for each class
        dice_score = 0
        for class_idx in range(num_classes):
            input_class = inputs[:, class_idx, ...]  # [B, H, W]
            target_class = targets_one_hot[:, class_idx, ...]  # [B, H, W]
            
            input_flat = input_class.contiguous().view(-1)
            target_flat = target_class.contiguous().view(-1)
            
            intersection = (input_flat * target_flat).sum()
            dice_class = (2. * intersection + smooth) / (
                input_flat.sum() + target_flat.sum() + smooth
            )
            dice_score += dice_class
        
        # Average over all classes
        dice_score = dice_score / num_classes
        return 1 - dice_score
