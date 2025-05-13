# blueVision

This directory contains scripts for creating and training a YOLOv8 model to detect electrical symbols in construction blueprints.

## Improved Model Training Pipeline

The updated training pipeline addresses false positive issues by implementing these improvements:

1. **Blueprint-Based Backgrounds**: Extracts sections from real blueprints for more realistic synthetic data.
2. **Manual Annotations**: Supports including manually annotated examples for better context learning.
3. **Enhanced Augmentation**: Uses blueprint-specific noise patterns and realistic transformations.
4. **Optimized Training Parameters**: Adjusts IoU thresholds, augmentation settings, and learning parameters.

## Steps to Train an Improved Model

### 1. Extract Symbol Templates (if not already done)
```bash
python train/extract_templates.py
```

### 2. Extract Blueprint Backgrounds
```bash
python train/extract_blueprint_backgrounds.py --blueprint-dir data/images --num-patches 200
```

### 3. Generate Synthetic Training Data with Blueprint Backgrounds
```bash
python train/generate_synthetic.py --count 500 --use-blueprint-bg --blueprint-bg-ratio 0.7
```

### 4. (Optional) Add Manual Annotations
1. Create small crops of blueprint sections containing symbols
2. Place them in `data/manual/images/`
3. Create corresponding YOLO format annotations in `data/manual/labels/`

Example annotation format (one line per symbol):
```
class_id x_center y_center width height
```
Where:
- `class_id`: 0 for EVSE, 1 for Panel, 2 for GFI
- Coordinates are normalized to image size (0-1 range)

### 5. Prepare Dataset for Training
```bash
python train/prepare_data.py
```

### 6. Train the Model
```bash
python train/train_yolo.py --epochs 100 --img-size 640
```

### 7. Run Detection with Optimized Settings
```bash
python detect.py --input data/images/your_blueprint.png --tile-size 640 --overlap 0.3 --nms 0.45
```

## Training Arguments

You can customize the training with the following parameters:

```bash
python train/train_yolo.py --help
```

Key parameters:
- `--epochs`: Number of training epochs (default: 100)
- `--img-size`: Input image size (default: 640)
- `--batch`: Batch size (default: 8)
- `--iou-thres`: IoU threshold for NMS (default: 0.7)
- `--conf-thres`: Confidence threshold (default: 0.25)

## Detection Arguments

The detection script supports the following parameters:

```bash
python detect.py --help
```

Key parameters:
- `--input`: Input blueprint file or directory
- `--tile-size`: Size of tiles for processing large blueprints (default: 640)
- `--overlap`: Overlap between tiles (default: 0.2)
- `--nms`: NMS threshold for merging overlapping detections (default: 0.5)
- `--conf`: Confidence threshold (default: 0.4)

## Recommended Workflow for Best Results

1. Start with a small set of high-quality templates (1-3 per class)
2. Generate synthetic data with blueprint backgrounds
3. Train the model with default parameters
4. Use the model to detect symbols on test blueprints
5. Manually annotate a few examples where the model fails
6. Add these manual annotations to your dataset
7. Retrain with the enhanced dataset

This "human-in-the-loop" approach progressively improves model accuracy with minimal manual effort.
