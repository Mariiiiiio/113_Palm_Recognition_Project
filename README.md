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
- Classes: [List of your palm classes]

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
- Training set: X images
- Validation set: Y images
- Test set: Z images
- Annotations in YOLO format
- Classes: [List your palm classes]

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