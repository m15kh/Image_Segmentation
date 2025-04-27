import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class Patching_Dataset(Dataset):
    def __init__(self, latent, input_row, input_col, step):
        self.input_row = input_row
        self.input_col = input_col
        shape_latent = latent.shape        
        ROW = shape_latent[0]
        COL = shape_latent[1]

        # Create row and column index ranges
        row_list_1 = range(input_row, ROW + 1, step)
        row_list_2 = range(ROW, row_list_1[-1] - 1, -step)
        row_list = [*row_list_1, *row_list_2]

        col_list_1 = range(input_col, COL + 1, step)
        col_list_2 = range(COL, col_list_1[-1] - 1, -step)
        col_list = [*col_list_1, *col_list_2]

        # Calculate the total number of patches
        self.num_patch = len(row_list) * len(col_list)

        # Store row and column indices
        row_col_inds = np.zeros([self.num_patch, 2]).astype(np.int32)
        # self.latent = latent.reshape(1, ROW, COL)
        self.latent = latent.reshape(ROW, COL)
        ind = 0
        for row_ind in row_list:
            for col_ind in col_list:
                row_col_inds[ind, :] = [row_ind, col_ind]
                ind += 1

        # Convert indices to PyTorch tensor
        self.row_col_inds = torch.from_numpy(row_col_inds)

    def __len__(self):
        return self.num_patch

    def __getitem__(self, ind):
        row_col = self.row_col_inds[ind].clone()
        row = row_col[0]
        col = row_col[1]

        # Extract the patch based on the row and column indices
        # patch = self.latent[:, (row - self.input_row):row, (col - self.input_col):col]
        # return row_col, patch
        # Extract the patch based on the row and column indices
        patch = self.latent[(row - self.input_row):row, (col - self.input_col):col]
        return row_col, patch