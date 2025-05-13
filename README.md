# BlueVision

Where electrical symbols hide, our pixels seek – faster than an electrician with X-ray goggles.

## Overview

BlueVision is a computer vision system built on YOLOv8 for detecting tiny electrical symbols in architectural blueprints. The detector is specifically trained to identify three types of electrical components:

- **EVSE (Electric Vehicle Supply Equipment)**: Charging stations for electric vehicles
- **Panel**: Electrical panels or distribution boards
- **GFI (Ground Fault Interrupter)**: Safety devices that protect against electrical shocks

The detector handles the challenges of detecting small symbols in large blueprint sheets through techniques like tiled inference, multi-scale detection, and test-time augmentation.

## Features

- **Multi-scale Detection**: Process images at different scales to catch symbols of varying sizes
- **Tiled Processing**: Efficiently process large blueprint images by breaking them into overlapping tiles
- **Interactive Web Interface**: Gradio-based UI for easy uploading and processing of blueprints
- **Command-line Interface**: Batch processing capabilities via the `detect.py` script

## Project Structure

```
.
├── app/                      # Application code
│   ├── detector.py           # Core detection module
│   ├── models/weights/       # Contains best.pt model file
│   ├── results/              # Detection results (created at runtime)
│   └── utils/                # Utility functions
├── data/                     # Data directory (structure created at runtime)
│   ├── images/               # Blueprint images
│   │   ├── train/            # Training images
│   │   └── val/              # Validation images
│   └── raw/                  # Raw PDF blueprints
├── train/                    # Training scripts
│   ├── annotate_full_blueprints.py  # Annotation tool
│   ├── extract_blueprints.py        # PDF to image extraction
│   ├── train_tiny_objects.py        # YOLOv8 training script
│   └── tiny_objects_hyp.yaml        # Hyperparameters for tiny objects
├── app.py                    # Gradio web interface
├── detect.py                 # Command-line detection script
├── blueprint_data.yaml       # Dataset configuration
└── requirements.txt          # Python dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VanshikWaghela/blueVision.git
   cd blueVision
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Model weights:
   - The model weights file `best.pt` should be in `app/models/weights/best.pt` 
   - If downloading separately, create the directory and place the file there

## Data Structure Setup

When running for the first time, the necessary directory structure will be created:

```bash
# Create data directories if they don't exist
mkdir -p data/images/train data/images/val data/raw
```

## Usage

### Web Interface

Run the Gradio web interface:

```bash
python app.py
```

This will launch a web server at http://localhost:7860 with an interactive interface.

### Command-line Detection

Process a single image or a directory of images:

```bash
# Process a single image with default parameters
python detect.py --input path/to/blueprint.png

# Process all images in a directory with custom parameters
python detect.py --input path/to/blueprints/ --conf 0.5 --scales 640,960 --output results/
```

### Training a New Model

1. Prepare your dataset by organizing blueprint images and annotations in the YOLO format:
   ```
   data/
   ├── images/
   │   ├── train/  # Contains training images (.jpg/.png)
   │   └── val/    # Contains validation images (.jpg/.png)
   └── labels/
       ├── train/  # Contains training labels (.txt)
       └── val/    # Contains validation labels (.txt)
   ```

2. Update `blueprint_data.yaml` to point to your dataset

3. Run the training script:
   ```bash
   python train/train_tiny_objects.py --data blueprint_data.yaml --epochs 300 --name my_symbol_detector
   ```

## Model Architecture

The detector is based on YOLOv8 (You Only Look Once) with the following customizations:

- Custom hyperparameter tuning for tiny object detection
- Enhanced augmentation strategies to handle the limited variability in blueprint data
- Non-maximum suppression (NMS) optimized for close-proximity symbols

## License

[MIT License](LICENSE)

## Author

Created by [Vanshik Waghela](https://github.com/VanshikWaghela)

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base detection framework
- [Gradio](https://gradio.app/) for the web interface components
