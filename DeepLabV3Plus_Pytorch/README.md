# DeepLabV3Plus-Pytorch

This repository is a fork of [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch.git), which implements the DeepLabV3+ model for semantic segmentation.

## Installation

To install the required dependencies, run the following command:

```bash
bash install.sh
```

## Using the Model with Custom Data

To use this repository with your custom dataset, follow these steps:

1. Prepare your dataset in the required format:
   - Images should be stored in a directory.
   - Corresponding segmentation masks should be stored in another directory.

2. Update the dataset configuration in the script you are using (e.g., `train.py` or `test.py`) to point to your custom dataset paths.

3. Run the training or testing script. For example, to train the model:

   ```bash
   python train.py --data-dir /path/to/your/dataset --save-dir /path/to/save/checkpoints
   ```

   Replace `/path/to/your/dataset` with the path to your dataset and `/path/to/save/checkpoints` with the directory where you want to save the model checkpoints.

## Reference

This repository is based on the original implementation by [VainF](https://github.com/VainF/DeepLabV3Plus-Pytorch.git). Please refer to the original repository for additional details and documentation.
