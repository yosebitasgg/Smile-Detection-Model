# Smile Detection Model

A neural network model for detecting smiles in facial images using PyTorch.

## Description

This project implements a smile detection classifier using the LFW (Labeled Faces in the Wild) dataset. The model classifies facial images as smiling or non-smiling.

## Dataset

This project uses the LFW cropped color dataset. The dataset is not included in this repository due to its size.

**To use this project:**
1. Download the LFW cropped color dataset
2. Place it in a folder named `lfwcrop_color/` in the project root

## Project Structure

```
├── dataset.py          # Dataset loader for LFW images
├── SMILE_list.txt      # List of smiling face images
├── NON-SMILE_list.txt  # List of non-smiling face images
└── README.md
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Pillow
- NumPy

## Usage

```python
from dataset import LFWDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

dataset = LFWDataset(
    faces_folder='lfwcrop_color/faces',
    smiling_labels_file='SMILE_list.txt',
    non_smiling_labels_file='NON-SMILE_list.txt',
    transform=transform
)
```

## License

This project is for educational purposes.
