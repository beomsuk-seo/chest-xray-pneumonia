# Deep Learning for Pneumonia Diagnosis from Chest X-Rays

### Project Overview

This project uses deep learning with TensorFlow/Keras to classify chest X-ray images into three categories: NORMAL, VIRAL PNEUMONIA, and BACTERIAL PNEUMONIA. The model is based on ResNet50 and employs transfer learning and fine-tuning to improve performance on a medical imaging dataset.

### Dataset Organization
- Original dataset contains all pneumonia imaging in a single folder.
- `src/split_pneumonia.py` splits them into `BACTERIAL/` and `VIRAL/` directores for three-class classification.

Initial Data Heirarchy
```
data/raw/
├── train/
│   ├── NORMAL/       # Training images, healthy lungs
│   └── PNEUMONIA/    # Training images, pneumonia
├── val/
│   ├── NORMAL/       # Validation set images, healthy lungs
│   └── PNEUMONIA/    # Validation set images, pneumonia
└── test/
    ├── NORMAL/        # Test set images, healthy lungs
    └── PNEUMONIA/    # Test set images, pneumonia
```
Processed Data Heirarchy
```
data/

WIP
```