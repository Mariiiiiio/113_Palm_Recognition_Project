# Palm Project

## Overview
This project is part of a research initiative supported by the National Science Council. The project focuses on palm-related research and analysis using various YOLO (You Only Look Once) model implementations.

## Project Structure
```
├── Finalize_program/    # Final implementation of the project
├── yolov9/             # YOLOv9 implementation
├── yolov10/            # YOLOv10 implementation
├── yolov11/            # YOLOv11 implementation
└── YoloV12/            # YOLOv12 implementation
```

## Data Structure
The project uses the following data structure for palm detection and analysis:

### Dataset Organization
```
dataset/
├── train/
│   ├── images/         # Training images
│   └── labels/         # Training annotations in YOLO format
├── valid/
│   ├── images/         # Validation images
│   └── labels/         # Validation annotations in YOLO format
└── test/
    ├── images/         # Test images
    └── labels/         # Test annotations in YOLO format
```

### Label Format
- Each image has a corresponding .txt file in YOLO format
- Format: `<class> <x_center> <y_center> <width> <height>`
- Coordinates are normalized to [0, 1]


## Getting Started

### Prerequisites
- Python 3.8+
- Required dependencies (listed in requirements.txt)
- CUDA-capable GPU (recommended)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/[your-username]/Palm-Project.git
cd Palm-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Program

### Training
```bash
# For YOLOv12 (latest version)
cd YoloV12
python train.py --data data.yaml --epochs 100 --batch-size 16 --img 640

# For other versions
cd yolov[version]
python train.py --data data.yaml --epochs 100 --batch-size 16 --img 640
```

### Inference
```bash
# For detection on images
python detect.py --source path/to/image --weights path/to/weights.pt

# For detection on video
python detect.py --source path/to/video --weights path/to/weights.pt
```

### Evaluation
```bash
python val.py --data data.yaml --weights path/to/weights.pt
```

## Dataset
Our palm dataset is available on Google Drive. You can access it using the following link:
[Palm Dataset](https://drive.google.com/drive/folders/your-dataset-folder-id)

The dataset includes:
- Training set: 558 images
- Validation set: 388 images
- Test set: 186 images
- Annotations in YOLO format

### Label Format
- Each image has a corresponding .txt file in YOLO format
- Format: `<class> <x_center> <y_center> <width> <height>`
- Coordinates are normalized to [0, 1]
- Total Classes: 104 (52 Left Palm and 52 Right Palm variations)
- Classes are organized as follows:
  
  Left Palm (0-51):
  ```
  0: Left_D1847392  | 13: Left_D2938471 | 26: Left_D3847291 | 39: Left_D4738291
  1: Left_D2938472  | 14: Left_D3847362 | 27: Left_D4738291 | 40: Left_D5847392
  2: Left_D3847563  | 15: Left_D4738291 | 28: Left_D5847392 | 41: Left_D6738291
  3: Left_D4738291  | 16: Left_D5847392 | 29: Left_D6738291 | 42: Left_D7847392
  4: Left_D5847392  | 17: Left_D6738291 | 30: Left_D7847392 | 43: Left_D8738291
  5: Left_D6738291  | 18: Left_D7847392 | 31: Left_D8738291 | 44: Left_D9847392
  6: Left_D7847392  | 19: Left_D8738291 | 32: Left_D9847392 | 45: Left_D1938472
  7: Left_D8738291  | 20: Left_D9847392 | 33: Left_D1938472 | 46: Left_D2847563
  8: Left_D9847392  | 21: Left_D1938472 | 34: Left_D2847563 | 47: Left_D3738291
  9: Left_D1938472  | 22: Left_D2847563 | 35: Left_D3738291 | 48: Left_D4847392
  10: Left_D2847563 | 23: Left_D3738291 | 36: Left_D4847392 | 49: Left_D5738291
  11: Left_D3738291 | 24: Left_D4847392 | 37: Left_D5738291 | 50: Left_D6847392
  12: Left_D4847392 | 25: Left_D5738291 | 38: Left_D6847392 | 51: Left_D7938472

  52: Right_D1847392 | 65: Right_D2938471 | 78: Right_D3847291 | 91: Right_D4738291
  53: Right_D2938472 | 66: Right_D3847362 | 79: Right_D4738291 | 92: Right_D5847392
  54: Right_D3847563 | 67: Right_D4738291 | 80: Right_D5847392 | 93: Right_D6738291
  55: Right_D4738291 | 68: Right_D5847392 | 81: Right_D6738291 | 94: Right_D7847392
  56: Right_D5847392 | 69: Right_D6738291 | 82: Right_D7847392 | 95: Right_D8738291
  57: Right_D6738291 | 70: Right_D7847392 | 83: Right_D8738291 | 96: Right_D9847392
  58: Right_D7847392 | 71: Right_D8738291 | 84: Right_D9847392 | 97: Right_D1938472
  59: Right_D8738291 | 72: Right_D9847392 | 85: Right_D1938472 | 98: Right_D2847563
  60: Right_D9847392 | 73: Right_D1938472 | 86: Right_D2847563 | 99: Right_D3738291
  61: Right_D1938472 | 74: Right_D2847563 | 87: Right_D3738291 | 100: Right_D4847392
  62: Right_D2847563 | 75: Right_D3738291 | 88: Right_D4847392 | 101: Right_D5738291
  63: Right_D3738291 | 76: Right_D4847392 | 89: Right_D5738291 | 102: Right_D6847392
  64: Right_D4847392 | 77: Right_D5738291 | 90: Right_D6847392 | 103: Right_D7938472
  ```


Please note: You need appropriate permissions to access the dataset. Contact the repository owner for access.

## Usage
Detailed usage instructions for each YOLO version implementation will be provided in their respective directories.

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- National Science Council for supporting this research
- Contributors and researchers involved in the project

## Contact
For any inquiries about this project, please open an issue in the GitHub repository. 
