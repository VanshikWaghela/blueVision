#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path
import random
import shutil
import argparse
import yaml

# Constants for cropping
CROP_SIZE = 640  # Size of the crop window
OVERLAP = 0.5    # Overlap between adjacent crops
MIN_SYMBOLS = 1  # Minimum number of symbols in a crop to be included
PADDING = 50     # Padding around symbols for context

def parse_args():
    parser = argparse.ArgumentParser(description="Generate crops from blueprints focusing on symbol regions")
    parser.add_argument("--blueprint-dir", type=str, default="data/images",
                      help="Directory containing blueprint images")
    parser.add_argument("--labels-dir", type=str, default="data/labels/train",
                      help="Directory containing YOLO format labels")
    parser.add_argument("--output-dir", type=str, default="data/crops",
                      help="Directory to save cropped images and labels")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                      help="Ratio of crops to use for training vs validation")
    parser.add_argument("--random-crops", type=int, default=20,
                      help="Number of additional random crops to generate per blueprint")
    parser.add_argument("--focused-crops", type=int, default=30,
                      help="Number of additional crops focused on symbols")
    parser.add_argument("--neighbor-crops", type=int, default=10,
                      help="Number of crops that are neighbors to symbol-containing crops")
    parser.add_argument("--use-sliding-window", action="store_true",
                      help="Use sliding window approach to ensure coverage")
    return parser.parse_args()

def read_yolo_labels(label_file):
    """Read YOLO format labels (class x_center y_center width height)"""
    if not Path(label_file).exists():
        return []
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append((class_id, x_center, y_center, width, height))
    
    return labels

def convert_yolo_to_absolute(label, img_width, img_height):
    """Convert normalized YOLO coordinates to absolute pixel coordinates"""
    class_id, x_center, y_center, width, height = label
    
    # Convert to absolute coordinates
    x_center = x_center * img_width
    y_center = y_center * img_height
    width = width * img_width
    height = height * img_height
    
    # Calculate bounding box corners
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return class_id, x1, y1, x2, y2

def convert_absolute_to_yolo(class_id, x1, y1, x2, y2, crop_width, crop_height):
    """Convert absolute coordinates to normalized YOLO format"""
    width = (x2 - x1) / crop_width
    height = (y2 - y1) / crop_height
    x_center = (x1 + x2) / 2 / crop_width
    y_center = (y1 + y2) / 2 / crop_height
    
    # Ensure values are within [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def get_crops_with_sliding_window(img_width, img_height, crop_size, overlap):
    """Generate crop coordinates using sliding window approach"""
    stride = int(crop_size * (1 - overlap))
    crops = []
    
    for y in range(0, img_height - crop_size + 1, stride):
        for x in range(0, img_width - crop_size + 1, stride):
            crops.append((x, y, x + crop_size, y + crop_size))
    
    # Add edge crops if needed
    if img_width % stride != 0:
        for y in range(0, img_height - crop_size + 1, stride):
            x = max(0, img_width - crop_size)
            crops.append((x, y, x + crop_size, y + crop_size))
    
    if img_height % stride != 0:
        for x in range(0, img_width - crop_size + 1, stride):
            y = max(0, img_height - crop_size)
            crops.append((x, y, x + crop_size, y + crop_size))
    
    # Add the bottom-right corner crop
    if img_width > crop_size and img_height > crop_size:
        crops.append((
            img_width - crop_size, 
            img_height - crop_size, 
            img_width, 
            img_height
        ))
    
    return crops

def get_symbol_focused_crops(labels, img_width, img_height, crop_size, num_crops, padding=PADDING):
    """Generate crops centered around symbols or groups of nearby symbols"""
    if not labels:
        return []
    
    # Convert YOLO to absolute coordinates
    absolute_boxes = [convert_yolo_to_absolute(label, img_width, img_height) for label in labels]
    
    # Get centers of all symbols
    centers = [(class_id, (x1 + x2) // 2, (y1 + y2) // 2) for class_id, x1, y1, x2, y2 in absolute_boxes]
    
    # Generate crops focused on individual symbols with padding
    focused_crops = []
    for _ in range(num_crops):
        # Pick a random symbol as anchor
        class_id, cx, cy = random.choice(centers)
        
        # Add random jitter to center point (but keep symbol within crop)
        jitter_x = random.randint(-crop_size//4, crop_size//4)
        jitter_y = random.randint(-crop_size//4, crop_size//4)
        
        # Calculate crop boundaries with the symbol roughly in center
        half_size = crop_size // 2
        x1 = max(0, cx - half_size + jitter_x)
        y1 = max(0, cy - half_size + jitter_y)
        
        # Adjust if crop goes beyond image bounds
        if x1 + crop_size > img_width:
            x1 = max(0, img_width - crop_size)
        if y1 + crop_size > img_height:
            y1 = max(0, img_height - crop_size)
        
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        
        focused_crops.append((x1, y1, x2, y2))
    
    return focused_crops

def get_neighbor_crops(base_crops, img_width, img_height, crop_size, num_crops):
    """Generate crops that are neighbors to existing crops"""
    if not base_crops:
        return []
    
    neighbor_crops = []
    for _ in range(num_crops):
        # Choose a random base crop
        x1, y1, x2, y2 = random.choice(base_crops)
        
        # Generate a neighboring crop with some overlap
        direction = random.choice(['left', 'right', 'up', 'down'])
        overlap = random.uniform(0.1, 0.5)
        
        if direction == 'left':
            shift = int(crop_size * (1 - overlap))
            nx1 = max(0, x1 - shift)
            ny1 = y1
            nx2 = nx1 + crop_size
            ny2 = ny1 + crop_size
        elif direction == 'right':
            shift = int(crop_size * (1 - overlap))
            nx1 = min(img_width - crop_size, x1 + shift)
            ny1 = y1
            nx2 = nx1 + crop_size
            ny2 = ny1 + crop_size
        elif direction == 'up':
            shift = int(crop_size * (1 - overlap))
            nx1 = x1
            ny1 = max(0, y1 - shift)
            nx2 = nx1 + crop_size
            ny2 = ny1 + crop_size
        else:  # down
            shift = int(crop_size * (1 - overlap))
            nx1 = x1
            ny1 = min(img_height - crop_size, y1 + shift)
            nx2 = nx1 + crop_size
            ny2 = ny1 + crop_size
            
        # Check if the crop is within bounds
        if nx1 >= 0 and ny1 >= 0 and nx2 <= img_width and ny2 <= img_height:
            neighbor_crops.append((nx1, ny1, nx2, ny2))
    
    return neighbor_crops

def get_random_crops(img_width, img_height, crop_size, num_crops):
    """Generate random crops from the image"""
    random_crops = []
    for _ in range(num_crops):
        # Ensure crop is within image bounds
        x1 = random.randint(0, max(0, img_width - crop_size))
        y1 = random.randint(0, max(0, img_height - crop_size))
        x2 = x1 + crop_size
        y2 = y1 + crop_size
        random_crops.append((x1, y1, x2, y2))
    
    return random_crops

def filter_labels_for_crop(labels, crop_x1, crop_y1, crop_x2, crop_y2, img_width, img_height):
    """Filter and adjust labels for a specific crop"""
    filtered_labels = []
    
    for label in labels:
        class_id, x_center, y_center, width, height = label
        
        # Convert to absolute coordinates
        abs_x_center = x_center * img_width
        abs_y_center = y_center * img_height
        abs_width = width * img_width
        abs_height = height * img_height
        
        # Calculate bounding box corners
        abs_x1 = abs_x_center - abs_width / 2
        abs_y1 = abs_y_center - abs_height / 2
        abs_x2 = abs_x_center + abs_width / 2
        abs_y2 = abs_y_center + abs_height / 2
        
        # Check if box is at least partially inside crop
        if (abs_x2 > crop_x1 and abs_x1 < crop_x2 and 
            abs_y2 > crop_y1 and abs_y1 < crop_y2):
            
            # Clip box to crop boundaries
            clipped_x1 = max(abs_x1, crop_x1)
            clipped_y1 = max(abs_y1, crop_y1)
            clipped_x2 = min(abs_x2, crop_x2)
            clipped_y2 = min(abs_y2, crop_y2)
            
            # Check if the clipped box still has reasonable area
            clipped_width = clipped_x2 - clipped_x1
            clipped_height = clipped_y2 - clipped_y1
            
            if clipped_width > 1 and clipped_height > 1:
                # Convert back to crop-relative YOLO format
                crop_width = crop_x2 - crop_x1
                crop_height = crop_y2 - crop_y1
                
                rel_clipped_x1 = (clipped_x1 - crop_x1) / crop_width
                rel_clipped_y1 = (clipped_y1 - crop_y1) / crop_height
                rel_clipped_width = clipped_width / crop_width
                rel_clipped_height = clipped_height / crop_height
                
                # Convert to YOLO center format
                rel_x_center = rel_clipped_x1 + rel_clipped_width / 2
                rel_y_center = rel_clipped_y1 + rel_clipped_height / 2
                
                filtered_labels.append((class_id, rel_x_center, rel_y_center, rel_clipped_width, rel_clipped_height))
    
    return filtered_labels

def process_blueprint(blueprint_path, labels_dir, output_dir, args, is_train=True):
    """Process a single blueprint with various cropping strategies"""
    blueprint_name = blueprint_path.stem
    label_path = Path(labels_dir) / f"{blueprint_name}.txt"
    
    # Read image and labels
    img = cv2.imread(str(blueprint_path))
    if img is None:
        print(f"Error reading image: {blueprint_path}")
        return 0
    
    img_height, img_width = img.shape[:2]
    labels = read_yolo_labels(label_path)
    if not labels:
        print(f"No labels found for: {blueprint_path}")
        return 0
    
    # Prepare output directories
    output_type = "train" if is_train else "val"
    output_img_dir = Path(output_dir) / "images" / output_type
    output_label_dir = Path(output_dir) / "labels" / output_type
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all types of crops
    all_crops = []
    
    # 1. Sliding window crops if enabled
    if args.use_sliding_window:
        sliding_crops = get_crops_with_sliding_window(img_width, img_height, CROP_SIZE, OVERLAP)
        all_crops.extend(sliding_crops)
    
    # 2. Symbol-focused crops
    focused_crops = get_symbol_focused_crops(labels, img_width, img_height, CROP_SIZE, args.focused_crops)
    all_crops.extend(focused_crops)
    
    # 3. Neighbor crops (to provide context)
    neighbor_crops = get_neighbor_crops(focused_crops, img_width, img_height, CROP_SIZE, args.neighbor_crops)
    all_crops.extend(neighbor_crops)
    
    # 4. Random crops (for background variety and hard negative mining)
    random_crops = get_random_crops(img_width, img_height, CROP_SIZE, args.random_crops)
    all_crops.extend(random_crops)
    
    # Remove duplicates and shuffle
    unique_crops = list(set(all_crops))
    random.shuffle(unique_crops)
    
    # Process each crop
    crop_count = 0
    for i, (x1, y1, x2, y2) in enumerate(unique_crops):
        # Extract crop image
        crop_img = img[y1:y2, x1:x2]
        
        # Filter and adjust labels for this crop
        crop_labels = filter_labels_for_crop(labels, x1, y1, x2, y2, img_width, img_height)
        
        # Only keep crops with symbols unless it's a random crop (for hard negatives)
        if len(crop_labels) >= MIN_SYMBOLS or (x1, y1, x2, y2) in random_crops:
            # Generate unique filename
            crop_filename = f"{blueprint_name}_crop_{i:04d}.png"
            label_filename = f"{blueprint_name}_crop_{i:04d}.txt"
            
            # Save crop image
            cv2.imwrite(str(output_img_dir / crop_filename), crop_img)
            
            # Save crop labels
            with open(output_label_dir / label_filename, 'w') as f:
                for label in crop_labels:
                    class_id, x_center, y_center, width, height = label
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            crop_count += 1
    
    return crop_count

def create_dataset_yaml(output_dir, class_names):
    """Create YAML configuration file for YOLOv8 training"""
    train_path = str(Path(output_dir) / "images/train")
    val_path = str(Path(output_dir) / "images/val")
    
    yaml_content = {
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": class_names
    }
    
    yaml_path = Path(output_dir) / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset configuration: {yaml_path}")
    return yaml_path

def main():
    args = parse_args()
    blueprint_dir = Path(args.blueprint_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    
    # Get blueprint images (png and jpg)
    blueprints = list(blueprint_dir.glob("**/*.png")) + list(blueprint_dir.glob("**/*.jpg"))
    if not blueprints:
        print(f"No blueprint images found in {blueprint_dir}")
        return
    
    print(f"Found {len(blueprints)} blueprint images")
    
    # Split into train and validation
    random.shuffle(blueprints)
    split_idx = int(len(blueprints) * args.train_ratio)
    train_blueprints = blueprints[:split_idx]
    val_blueprints = blueprints[split_idx:]
    
    # Process each blueprint
    train_count = 0
    for blueprint in train_blueprints:
        count = process_blueprint(blueprint, labels_dir, output_dir, args, is_train=True)
        train_count += count
        print(f"Generated {count} crops from {blueprint.name} for training")
    
    val_count = 0
    for blueprint in val_blueprints:
        count = process_blueprint(blueprint, labels_dir, output_dir, args, is_train=False)
        val_count += count
        print(f"Generated {count} crops from {blueprint.name} for validation")
    
    print(f"Total crops generated: {train_count + val_count} (Train: {train_count}, Val: {val_count})")
    
    # Create YAML file for training
    class_names = []
    class_file = Path("data/classes.txt")
    if class_file.exists():
        with open(class_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = ['evse', 'panel', 'gfi']  # Default classes
    
    yaml_path = create_dataset_yaml(output_dir, class_names)
    
    print("\nDataset preparation complete!")
    print(f"To train on this dataset, use: python train/train_optimized.py --data {yaml_path}")

if __name__ == "__main__":
    main() 