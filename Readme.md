# U-Net and AutoEncoder Repository

## Overview

This repository contains implementations for:

1. **U-Net**: A semantic segmentation model with training and inference capabilities.
2. **U-Net_v2**: An improved version of U-Net with a better structure, recommended for use.
3. **AutoEncoder**: Includes both vanilla and CNN-based autoencoder implementations.

## Structure

```
/U_NET/
├── U_Net/               # Original U-Net implementation
│   ├── models/          # Model architecture, training, and inference scripts
│   └── data/            # Dataset for training and testing
├── U_Net_v2/            # Improved U-Net implementation
│   ├── scripts/         # Training, inference, and preprocessing scripts
│   └── params/          # Configuration files for training and inference
├── Auto_Encoder/        # AutoEncoder implementations
│   ├── vanilla_autoencoder/  # Vanilla autoencoder
│   └── cnn-autoencoder/      # CNN-based autoencoder
└── Readme.md            # Repository documentation
```

## Recommendations

- **For Semantic Segmentation**: Use `U_Net_v2` as it has a better structure and improved functionality compared to the original `U_Net`.
- **For AutoEncoding Tasks**: Explore the `Auto_Encoder` folder for vanilla and CNN-based autoencoder implementations.

## Usage

### U-Net_v2

- **Training**: Use the `train.py` script in `U_Net_v2/scripts/src/`.
- **Inference**: Use the `inference.py` script in `U_Net_v2/scripts/src/`.

### AutoEncoder

- **Vanilla AutoEncoder**: Refer to the `vanilla_autoencoder` folder.
- **CNN AutoEncoder**: Refer to the `cnn-autoencoder` folder.

## License

[Specify license information]

## Acknowledgements

- U-Net architecture: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- [Any other acknowledgements]
