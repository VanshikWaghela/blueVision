#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 training optimized for tiny electrical symbols")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model to start from (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--name", type=str, default="tiny_symbol_detector", help="Name for the training run")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--device", type=str, default="", help="Device to train on (cuda device, i.e. 0 or 0,1,2,3 or cpu)")
    return parser.parse_args()

def verify_dataset(data_yaml_path):
    """Verify that the dataset exists and has the expected structure"""
    data_yaml_path = Path(data_yaml_path).resolve()
    print(f"Verifying dataset at: {data_yaml_path}")
    
    if not data_yaml_path.exists():
        print(f"Error: Data YAML file not found at {data_yaml_path}")
        return False
    
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Check required fields
        required_fields = ['train', 'val', 'nc', 'names']
        for field in required_fields:
            if field not in data_config:
                print(f"Error: Missing required field '{field}' in data config")
                return False
        
        # Check if directories exist
        train_dir = Path(data_config.get('train', ''))
        val_dir = Path(data_config.get('val', ''))
        
        if not train_dir.exists():
            print(f"Error: Train directory does not exist: {train_dir}")
            return False
            
        if not val_dir.exists():
            print(f"Error: Val directory does not exist: {val_dir}")
            return False
            
        # Count images in directories
        train_images = len(list(train_dir.glob('*.jpg'))) + len(list(train_dir.glob('*.png')))
        val_images = len(list(val_dir.glob('*.jpg'))) + len(list(val_dir.glob('*.png')))
        
        print(f"Found {train_images} training images and {val_images} validation images")
        
        if train_images == 0:
            print("Error: No training images found")
            return False
            
        if val_images == 0:
            print("Error: No validation images found")
            return False
        
        # Check labels directories
        train_label_dir = train_dir.parent.parent / "labels" / train_dir.name
        val_label_dir = val_dir.parent.parent / "labels" / val_dir.name
        
        if not train_label_dir.exists():
            print(f"Error: Train labels directory does not exist: {train_label_dir}")
            return False
            
        if not val_label_dir.exists():
            print(f"Error: Val labels directory does not exist: {val_label_dir}")
            return False
        
        # Count label files
        train_labels = len(list(train_label_dir.glob('*.txt')))
        val_labels = len(list(val_label_dir.glob('*.txt')))
        
        print(f"Found {train_labels} training label files and {val_labels} validation label files")
        
        if train_labels == 0:
            print("Error: No training label files found")
            return False
            
        if val_labels == 0:
            print("Error: No validation label files found")
            return False
        
        # Success
        print("Dataset verification successful!")
        return True
        
    except Exception as e:
        print(f"Error during dataset verification: {e}")
        return False

def main():
    args = parse_args()
    
    # Verify that the dataset exists and has the expected structure
    if not verify_dataset(args.data):
        print("Dataset verification failed. Please check your dataset structure.")
        sys.exit(1)
    
    # Load the model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Define training arguments optimized for tiny objects
    train_args = {
        # Dataset arguments
        "data": args.data,                  # Path to data.yaml file
        
        # Training arguments
        "epochs": args.epochs,              # Number of epochs
        "patience": 50,                     # Early stopping patience
        "batch": args.batch,                # Batch size
        "imgsz": args.imgsz,                # Image size
        "device": args.device,              # Device to use
        "project": "runs/train",            # Save to project/name
        "name": args.name,                  # Name of training run
        "pretrained": True,                 # Use pretrained weights
        "resume": args.resume,              # Resume training from last checkpoint
        
        # Loss function arguments - prioritize box accuracy for tiny objects
        "box": 10.0,                        # Box loss gain (higher to focus on accurate boxes)
        "cls": 0.5,                         # Class loss gain
        "dfl": 2.0,                         # Distribution focal loss gain
        
        # Augmentation arguments - heavy augmentation for small dataset
        "hsv_h": 0.015,                     # Image HSV-Hue augmentation
        "hsv_s": 0.7,                       # Image HSV-Saturation augmentation - stronger
        "hsv_v": 0.4,                       # Image HSV-Value augmentation
        "degrees": 15.0,                    # Image rotation (+/- deg) - moderate
        "translate": 0.2,                   # Image translation (+/- fraction)
        "scale": 0.5,                       # Image scale (+/- gain) - stronger scaling
        "shear": 10.0,                      # Image shear (+/- deg)
        "perspective": 0.0005,              # Image perspective (+/- fraction) - minimal distortion
        "flipud": 0.5,                      # Image flip up-down (probability) - higher
        "fliplr": 0.5,                      # Image flip left-right (probability)
        "mosaic": 1.0,                      # Image mosaic (probability) - maximum
        "mixup": 0.3,                       # Image mixup (probability) - considerable
        "copy_paste": 0.3,                  # Segment copy-paste (probability)
        
        # Logging and visualization
        "plots": True,                      # Save plots during train/val
        "save": True,                       # Save checkpoints
    }
    
    # Add hyperparameters specific to tiny object detection
    if not args.resume:
        # Custom hyperparameters for tiny object detection can be defined in a YAML file
        hyperparam_file = Path("train/tiny_objects_hyp.yaml")
        if hyperparam_file.exists():
            with open(hyperparam_file, 'r') as f:
                custom_hyp = yaml.safe_load(f)
                # Update train_args with custom hyperparameters
                train_args.update(custom_hyp)
                print(f"Loaded custom hyperparameters from {hyperparam_file}")
    
    # Print training configuration
    print("\n=== YOLOv8 Training for Tiny Electrical Symbols ===")
    print(f"Starting training with {args.model} for {args.epochs} epochs")
    print(f"Using enhanced augmentation optimized for tiny objects")
    print(f"Image size: {args.imgsz}x{args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Data: {args.data}")
    print(f"Results will be saved to: runs/train/{args.name}")
    
    try:
        # Start training
        results = model.train(**train_args)
        
        # Print training results
        print("\n=== Training Complete ===")
        metrics = results.results_dict
        print(f"Best fitness: {metrics.get('fitness', 'N/A')}")
        print(f"Final mAP@0.5: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Final mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        
        # Get path to best weights
        best_weights = Path('runs/train') / args.name / 'weights/best.pt'
        if best_weights.exists():
            print(f"\nBest model weights saved at: {best_weights}")
            print("\nTo perform detection with your trained model:")
            print(f"python detect.py --model {best_weights} --conf 0.4 --iou 0.5")
        else:
            last_weights = Path('runs/train') / args.name / 'weights/last.pt'
            print(f"\nLast model weights saved at: {last_weights}")
            print("\nTo perform detection with your trained model:")
            print(f"python detect.py --model {last_weights} --conf 0.4 --iou 0.5")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if the dataset structure is correct")
        print("2. Try reducing batch size if you encounter CUDA memory errors")
        print("3. Check for corrupted images or label files")
        print("4. Try with minimal parameters: python train/train_tiny_objects.py --data your_data.yaml --model yolov8n.pt --epochs 100")
        sys.exit(1)

if __name__ == "__main__":
    main() 