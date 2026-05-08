# Augmentation

Image augmentation assets and pipeline configurations used during data preprocessing.

## Augmentation Methods

TrOCR uses two data augmentation strategies:

### DA2 (Default Augmentation)
Used for IAM and SROIE datasets. Includes:
- Random rotation
- Random perspective/affine transforms
- Color jitter (brightness, contrast, saturation)
- Gaussian blur
- Image resizing to 384×384
- Normalization (mean=0.5, std=0.5)

### RandAugment
Used for STR Benchmark datasets. Includes:
- Random selection from a pool of augmentation operations
- Configurable magnitude and number of augmentations
- Probability of keeping the image intact (`intact_prob=0.5`)

## Assets

This directory may contain:
- Custom fonts for synthetic data generation
- Background textures for augmented image compositing
- Augmentation configuration files

## Reference

See `src/data_aug.py` for the full augmentation pipeline implementation.
