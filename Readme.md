# Auto-encoder and U-Net

## Overview
This project implements two neural network architectures: Auto-encoder and U-Net. Auto-encoders are used for unsupervised learning, focusing on encoding input data into a compressed representation and reconstructing the original data. U-Net is a convolutional network architecture designed for image segmentation tasks.

## Features
- Auto-encoder:
   - Dimensionality reduction
   - Data reconstruction
   - Customizable architecture
- U-Net:
   - Image segmentation
   - Encoder-decoder structure
   - Skip connections for better feature preservation

## Installation
1. Clone the repository:
    ```bash
    git clone <repository-url> 
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Auto-encoder
1. Train the Auto-encoder:
    ```bash
    python auto_encoder/train.py
    ```
2. Evaluate the Auto-encoder:
    ```bash
    python auto_encoder/evaluate.py
    ```

### U-Net
1. Train the U-Net:
    ```bash
    python unet/train.py
    ```
2. Evaluate the U-Net:
    ```bash
    python unet/evaluate.py
    ```

## File Structure
- `auto_encoder/`: Contains scripts and models for the Auto-encoder.
   - `train.py`: Script for training the Auto-encoder.
   - `evaluate.py`: Script for evaluating the Auto-encoder.
   - `models/`: Contains Auto-encoder model definitions.
   - `data/`: Directory for Auto-encoder input data.
- `unet/`: Contains scripts and models for the U-Net.
   - `train.py`: Script for training the U-Net.
   - `evaluate.py`: Script for evaluating the U-Net.
   - `models/`: Contains U-Net model definitions.
   - `data/`: Directory for U-Net input data.

## License
This project is licensed under the MIT License.
