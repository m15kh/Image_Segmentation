# U-Net Segmentation Model

## Overview

This repository contains implementation of a U-Net architecture for semantic segmentation tasks. The model is designed to process images and produce segmentation masks for character recognition, specifically identifying characters like "e" and "h" in images.

## Project Structure

```
/Auto_Encoder/U_Net/
├── data/
│   ├── json/            # JSON annotation files with polygon coordinates
│   └── test_data/       # Test dataset
├── [model files]        # U-Net model implementation
└── [other files]        # Additional scripts and utilities
```

## Dataset

The dataset consists of images and corresponding JSON annotation files. Each JSON file contains polygon coordinates that define the boundaries of characters in the images. The annotations include:

- Character labels (e.g., "e", "h")
- Point coordinates forming polygons
- Additional metadata

Example of annotation structure:
```json
{
  "label": "e",
  "points": [
    [x1, y1],
    [x2, y2],
    ...
  ],
  "group_id": null,
  "description": "",
  "shape_type": "polygon",
  "flags": {},
  "mask": null
}
```

## Model Architecture

The project implements the U-Net architecture, which consists of:

1. **Contracting Path (Encoder)**: Series of convolutional layers and max pooling operations that extract features from the input image
2. **Expansive Path (Decoder)**: Series of up-convolutions and concatenations with features from the contracting path
3. **Output Layer**: Final convolutional layer that produces the segmentation mask

## Requirements

- Python 3.x
- PyTorch
- NumPy
- OpenCV
- Additional dependencies (specify in requirements.txt)

## Usage

### Training

```bash
# Command for training the model
python train.py --data_path data/json --epochs 100 --batch_size 16
```

### Inference

```bash
# Command for running inference on new images
python predict.py --model_path models/trained_model.pth --input_path path/to/image.png
```

## Results

The model achieves [add performance metrics] on the test dataset.

## License

[Specify license information]

## Acknowledgements

- U-Net architecture: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- [Any other acknowledgements]
