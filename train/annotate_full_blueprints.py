#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path
import sys
import shutil
import uuid  # Added for unique ID generation
import time  # Added for timestamps
import datetime  # Added for readable timestamps
import glob  # For file pattern matching

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directory where the script will LOOK for original blueprint images to annotate
SOURCE_IMAGES_DIR = PROJECT_ROOT / "data" / "images"

# Output directories for train/val split
OUTPUT_DIR = PROJECT_ROOT / "data"
TRAIN_IMAGES_DIR = OUTPUT_DIR / "images" / "train"
VAL_IMAGES_DIR = OUTPUT_DIR / "images" / "val"
TRAIN_LABELS_DIR = OUTPUT_DIR / "labels" / "train"
VAL_LABELS_DIR = OUTPUT_DIR / "labels" / "val"

# New directories for storing extracted symbol crops
SYMBOL_CROPS_DIR = OUTPUT_DIR / "symbol_crops"
TRAIN_CROPS_DIR = SYMBOL_CROPS_DIR / "train"
VAL_CROPS_DIR = SYMBOL_CROPS_DIR / "val"

# Path for the blueprint_data.yaml file, to be saved in the project root
YAML_PATH = PROJECT_ROOT / "blueprint_data.yaml"

# Make sure the output directories exist
for dir_path in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR, 
                TRAIN_CROPS_DIR, VAL_CROPS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Global variables for mouse callback
drawing = False
ix, iy = -1, -1
rect_x, rect_y, rect_w, rect_h = 0, 0, 0, 0
img_clone = None
selection_made = False
zoom_factor = 1.0
zoom_pos_x, zoom_pos_y = 0, 0
drag_start_x, drag_start_y = -1, -1
dragging = False
current_class = None
PREVIEW_ANNOTATIONS = True  # Flag to enable/disable preview of annotations

# Classes we want to detect
classes = ["evse", "panel", "gfi"]

# Colors for each class (BGR format for OpenCV)
class_colors = {
    "evse": (0, 255, 0),    # Green
    "panel": (0, 0, 255),   # Red
    "gfi": (255, 0, 0)      # Blue  
}

# Dictionary to keep track of annotations for the current image
current_annotations = []

def print_status(img_path):
    """Print current annotation status and controls to terminal"""
    global current_annotations
    
    # Count annotations by class
    class_counts = {cls: 0 for cls in classes}
    for anno in current_annotations:
        cls_idx = int(anno.split()[0])
        if 0 <= cls_idx < len(classes):
            class_counts[classes[cls_idx]] += 1
    
    print("\n" + "="*50)
    print(f"Current image: {img_path.name}")
    print("="*50)
    print("CURRENT ANNOTATION COUNTS:")
    for cls in classes:
        print(f"  {cls}: {class_counts[cls]}")
    
    print("\nCONTROLS:")
    print("  Mouse WHEEL or Z/X keys: Zoom in/out")
    print("  MIDDLE BUTTON + drag: Pan around")
    print("  0, 1, 2 keys: Select class (evse, panel, gfi)")
    if current_class is not None:
        print(f"  CURRENTLY SELECTED CLASS: {classes[current_class]}")
    print("  LEFT CLICK + drag: Draw selection box")
    print("  C: Confirm selection")
    print("  R: Retry selection")
    print("  P: Toggle preview of annotations")
    print("  S: Save annotations to train/val sets")
    print("  D: Delete the last annotation")
    print("  N: Next image")
    print("  Q: Quit")
    print("-"*50)

def draw_rectangle(event, x, y, flags, param):
    """Mouse callback function to draw rectangle interactively and handle zooming"""
    global ix, iy, drawing, rect_x, rect_y, rect_w, rect_h, img_clone, selection_made
    global zoom_factor, zoom_pos_x, zoom_pos_y, drag_start_x, drag_start_y, dragging
    
    # Get the original image and its dimensions from param
    img_orig, display_scale = param
    h, w = img_orig.shape[:2]
    
    # Convert screen coordinates to image coordinates based on current zoom
    img_x = int((x / display_scale + zoom_pos_x) / zoom_factor)
    img_y = int((y / display_scale + zoom_pos_y) / zoom_factor)
    
    # Ensure coordinates are within image bounds
    img_x = max(0, min(img_x, w-1))
    img_y = max(0, min(img_y, h-1))
    
    if event == cv2.EVENT_MOUSEWHEEL:
        # Zoom in/out with mouse wheel
        if flags > 0:  # Scroll up, zoom in
            zoom_factor = min(5.0, zoom_factor * 1.2)
        else:  # Scroll down, zoom out
            zoom_factor = max(1.0, zoom_factor / 1.2)
        
        # Adjust zoom position to keep mouse point fixed
        zoom_pos_x = max(0, min(img_x * zoom_factor - x / display_scale, w * zoom_factor - w))
        zoom_pos_y = max(0, min(img_y * zoom_factor - y / display_scale, h * zoom_factor - h))
        
        # Update display
        update_display(img_orig, display_scale)
        print(f"Zoom level: {zoom_factor:.1f}x")
    
    elif event == cv2.EVENT_MBUTTONDOWN:
        # Middle button for panning
        drag_start_x = x
        drag_start_y = y
        dragging = True
    
    elif event == cv2.EVENT_MBUTTONUP:
        dragging = False
    
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Pan the view with middle mouse button
        dx = (x - drag_start_x) / display_scale
        dy = (y - drag_start_y) / display_scale
        
        zoom_pos_x = max(0, min(zoom_pos_x - dx, w * zoom_factor - w))
        zoom_pos_y = max(0, min(zoom_pos_y - dy, h * zoom_factor - h))
        
        drag_start_x = x
        drag_start_y = y
        
        # Update display
        update_display(img_orig, display_scale)
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing rectangle
        drawing = True
        selection_made = False
        ix, iy = img_x, img_y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw rectangle while dragging, but on a clean copy of the current view
            # to avoid accumulating multiple rectangles
            temp_display = np.array(img_clone).copy()
            
            # Calculate rectangle in original image space
            start_x = min(ix, img_x)
            start_y = min(iy, img_y)
            end_x = max(ix, img_x)
            end_y = max(iy, img_y)
            
            # Convert to screen space for display
            screen_start_x = int((start_x * zoom_factor - zoom_pos_x) * display_scale)
            screen_start_y = int((start_y * zoom_factor - zoom_pos_y) * display_scale)
            screen_end_x = int((end_x * zoom_factor - zoom_pos_x) * display_scale)
            screen_end_y = int((end_y * zoom_factor - zoom_pos_y) * display_scale)
            
            # Draw rectangle on the zoomed view
            cv2.rectangle(temp_display, (screen_start_x, screen_start_y), 
                         (screen_end_x, screen_end_y), (0, 255, 0), 2)
            cv2.imshow("Blueprint Image", temp_display)
    
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            selection_made = True
            
            # Calculate rectangle in original image space
            rect_x = min(ix, img_x)
            rect_y = min(iy, img_y)
            rect_w = abs(img_x - ix)
            rect_h = abs(img_y - iy)
            
            # Convert to screen space for display
            screen_rect_x = int((rect_x * zoom_factor - zoom_pos_x) * display_scale)
            screen_rect_y = int((rect_y * zoom_factor - zoom_pos_y) * display_scale)
            screen_rect_w = int(rect_w * zoom_factor * display_scale)
            screen_rect_h = int(rect_h * zoom_factor * display_scale)
            
            # Draw rectangle on the zoomed view (on a clean copy)
            temp_display = np.array(img_clone).copy()
            cv2.rectangle(temp_display, (screen_rect_x, screen_rect_y), 
                         (screen_rect_x + screen_rect_w, screen_rect_y + screen_rect_h), (0, 255, 0), 2)
            cv2.imshow("Blueprint Image", temp_display)
            
            print(f"Selected area: x={rect_x}, y={rect_y}, w={rect_w}, h={rect_h}")

def update_display(img_orig, display_scale):
    """Update the display with current zoom and position, optionally showing annotations"""
    global img_clone, current_annotations, PREVIEW_ANNOTATIONS
    
    h, w = img_orig.shape[:2]
    
    # Calculate visible area in original image
    view_x = int(zoom_pos_x / zoom_factor)
    view_y = int(zoom_pos_y / zoom_factor)
    view_w = int(w / zoom_factor)
    view_h = int(h / zoom_factor)
    
    # Ensure view is within bounds
    view_x = max(0, min(view_x, w - view_w))
    view_y = max(0, min(view_y, h - view_h))
    
    # Extract visible portion of original image
    visible_img = img_orig[view_y:view_y+view_h, view_x:view_x+view_w].copy()
    
    # Draw existing annotations if preview is enabled
    if PREVIEW_ANNOTATIONS and current_annotations:
        # Create a copy of the visible portion for drawing
        annotated_img = visible_img.copy()
        
        # Draw each annotation on the image
        for anno in current_annotations:
            # Parse the annotation
            parts = anno.split()
            cls_idx = int(parts[0])
            cls_name = classes[cls_idx]
            color = class_colors[cls_name]
            
            # Get normalized coordinates
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # Convert normalized coordinates to absolute coordinates in the original image
            abs_x_center = int(x_center * w)
            abs_y_center = int(y_center * h)
            abs_width = int(width * w)
            abs_height = int(height * h)
            
            # Calculate bounding box corners in original image space
            x1 = abs_x_center - abs_width // 2
            y1 = abs_y_center - abs_height // 2
            x2 = x1 + abs_width
            y2 = y1 + abs_height
            
            # Check if the box is visible in the current view
            if (x2 >= view_x and x1 < view_x + view_w and
                y2 >= view_y and y1 < view_y + view_h):
                
                # Adjust coordinates to the visible portion
                vis_x1 = max(0, x1 - view_x)
                vis_y1 = max(0, y1 - view_y)
                vis_x2 = min(view_w, x2 - view_x)
                vis_y2 = min(view_h, y2 - view_y)
                
                # Draw the rectangle
                cv2.rectangle(annotated_img, (vis_x1, vis_y1), (vis_x2, vis_y2), color, 2)
                
                # Draw label
                label_text = cls_name
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(annotated_img, (vis_x1, vis_y1 - text_height - 5), 
                              (vis_x1 + text_width + 5, vis_y1), color, -1)
                
                # Draw label text
                cv2.putText(annotated_img, label_text, (vis_x1 + 2, vis_y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        visible_img = annotated_img
    
    # Resize to screen size
    screen_w = int(view_w * zoom_factor * display_scale)
    screen_h = int(view_h * zoom_factor * display_scale)
    
    display_img = cv2.resize(visible_img, (screen_w, screen_h))
    
    # Store the clean image for future drawing
    img_clone = display_img.copy()
    cv2.imshow("Blueprint Image", img_clone)

def load_existing_annotations(annotation_path):
    """Load existing annotations if any"""
    global current_annotations
    
    current_annotations = []
    if annotation_path.exists():
        with open(annotation_path, "r") as f:
            current_annotations = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(current_annotations)} existing annotations from {annotation_path.name}")

def annotate_blueprints_interactive():
    """Interactive tool to annotate symbols directly on the full blueprint images"""
    global img_clone, rect_x, rect_y, rect_w, rect_h, selection_made
    global zoom_factor, zoom_pos_x, zoom_pos_y, current_class, current_annotations
    global PREVIEW_ANNOTATIONS
    
    # Find the blueprint images in data/images
    print(f"Searching for blueprint images in: {SOURCE_IMAGES_DIR.resolve()}")
    blueprint_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        blueprint_files.extend(list(SOURCE_IMAGES_DIR.glob(ext)))
    
    # Check if we found any files
    if not blueprint_files:
        print(f"Error: No blueprint images found in {SOURCE_IMAGES_DIR.resolve()}.")
        print("Please add some blueprint images to this directory first.")
        sys.exit(1)
    
    print(f"Found {len(blueprint_files)} blueprint images to annotate:")
    for idx, img_path in enumerate(blueprint_files):
        print(f"{idx+1}. {img_path.name}")
    
    # Create window and set mouse callback
    cv2.namedWindow("Blueprint Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Blueprint Image", 1400, 800)
    
    # Process each blueprint
    img_idx = 0
    while img_idx < len(blueprint_files):
        img_path = blueprint_files[img_idx]
        print(f"\nLoading {img_path.name} (image {img_idx+1}/{len(blueprint_files)})...")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path}, skipping...")
            img_idx += 1
            continue
        
        # Create/open annotation file path
        annotation_path = img_path.with_suffix(".txt")
        
        # Load existing annotations if any
        load_existing_annotations(annotation_path)
        
        # Calculate display scale for the window
        h, w = img.shape[:2]
        display_scale = min(1.0, 1400 / w, 800 / h)
        
        # Reset zoom for each new image
        zoom_factor = 1.0
        zoom_pos_x, zoom_pos_y = 0, 0
        current_class = None
        
        # Set the mouse callback with image and display scale as parameters
        cv2.setMouseCallback("Blueprint Image", draw_rectangle, param=(img, display_scale))
        
        # Initial display with clean image
        update_display(img, display_scale)
        
        # Print status and controls to terminal
        print_status(img_path)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('n'):  # Next image
                if current_annotations:
                    save_prompt = input("You have unsaved annotations. Save before moving to next image? (y/n): ").lower()
                    if save_prompt == 'y':
                        save_annotations_to_file(annotation_path)
                img_idx += 1
                break
                
            elif key == ord('p'):  # Previous image
                if img_idx > 0:
                    if current_annotations:
                        save_prompt = input("You have unsaved annotations. Save before moving to previous image? (y/n): ").lower()
                        if save_prompt == 'y':
                            save_annotations_to_file(annotation_path)
                    img_idx -= 1
                    break
                else:
                    print("Already at the first image")
                
            elif key == ord('q'):  # Quit
                if current_annotations:
                    save_prompt = input("You have unsaved annotations. Save before quitting? (y/n): ").lower()
                    if save_prompt == 'y':
                        save_annotations_to_file(annotation_path)
                cv2.destroyAllWindows()
                return
                
            elif key == ord('s'):  # Save annotations
                save_annotations_to_file(annotation_path)
                copy_to_dataset(img_path, img)
                
            elif key == ord('p'):  # Toggle preview
                PREVIEW_ANNOTATIONS = not PREVIEW_ANNOTATIONS
                print(f"Annotation preview: {'ON' if PREVIEW_ANNOTATIONS else 'OFF'}")
                update_display(img, display_scale)
                
            elif key == ord('d'):  # Delete last annotation
                if current_annotations:
                    deleted = current_annotations.pop()
                    print(f"Deleted annotation: {deleted}")
                    update_display(img, display_scale)
                    print_status(img_path)
                else:
                    print("No annotations to delete")
                
            elif key == ord('z'):  # Zoom in
                zoom_factor = min(5.0, zoom_factor * 1.2)
                update_display(img, display_scale)
                print(f"Zoomed in to {zoom_factor:.1f}x")
                
            elif key == ord('x'):  # Zoom out
                zoom_factor = max(1.0, zoom_factor / 1.2)
                update_display(img, display_scale)
                print(f"Zoomed out to {zoom_factor:.1f}x")
                
            elif key in [ord('0'), ord('1'), ord('2')]:  # Class selection
                cls_idx = key - ord('0')
                current_class = cls_idx
                cls_name = classes[cls_idx]
                print(f"Selected class: {cls_name}")
                
                # Reset selection variables
                selection_made = False
                rect_x, rect_y, rect_w, rect_h = 0, 0, 0, 0
                
                # Wait for user to draw rectangle
                print("LEFT CLICK and DRAG to select the symbol area")
                
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    
                    if key == ord('c') and selection_made and rect_w > 0 and rect_h > 0:  # Confirm selection
                        if current_class is None:
                            print("Please select a class first (0=evse, 1=panel, 2=gfi)")
                            break
                            
                        # Extract the region from the original image for preview
                        extracted = img[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
                        
                        # Create YOLO format annotation
                        # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                        # All values are normalized to [0, 1]
                        x_center = (rect_x + rect_w / 2) / w
                        y_center = (rect_y + rect_h / 2) / h
                        norm_width = rect_w / w
                        norm_height = rect_h / h
                        
                        # Add annotation to the list
                        annotation = f"{current_class} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                        current_annotations.append(annotation)
                        
                        # Show extracted region in a properly sized window
                        display_size = max(200, min(600, rect_w * 2, rect_h * 2))
                        cv2.namedWindow("Selected Symbol", cv2.WINDOW_NORMAL)
                        cv2.resizeWindow("Selected Symbol", display_size, display_size)
                        cv2.imshow("Selected Symbol", extracted)
                        
                        print(f"Added annotation: {annotation}")
                        
                        # Update main display with clean image
                        update_display(img, display_scale)
                        
                        # Print updated status
                        print_status(img_path)
                        break
                        
                    elif key == ord('r'):  # Retry selection
                        update_display(img, display_scale)
                        selection_made = False
                        rect_x, rect_y, rect_w, rect_h = 0, 0, 0, 0
                        print("Retrying selection")
                        break
                    
                    elif key == ord('b'):  # Back to class selection
                        update_display(img, display_scale)
                        current_class = None
                        print("Cancelled. Please select a class (0=evse, 1=panel, 2=gfi)")
                        break
        
        # Close any open windows for this image
        if cv2.getWindowProperty("Selected Symbol", cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow("Selected Symbol")
    
    cv2.destroyAllWindows()
    print("Annotation complete.")

def save_annotations_to_file(annotation_path):
    """Save annotations to a file"""
    global current_annotations
    
    # Save all annotations to file
    with open(annotation_path, "w") as f:
        for anno in current_annotations:
            f.write(f"{anno}\n")
    print(f"Saved {len(current_annotations)} annotations to {annotation_path}")
    return True

def extract_symbol_crops(img, annotations, dataset_type="train"):
    """Extract each annotated symbol as a separate image"""
    h, w = img.shape[:2]
    crops_dir = TRAIN_CROPS_DIR if dataset_type == "train" else VAL_CROPS_DIR
    
    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create list to store crop paths
    crop_paths = []
    
    # Extract each symbol
    for idx, anno in enumerate(annotations):
        parts = anno.split()
        cls_idx = int(parts[0])
        cls_name = classes[cls_idx]
        
        # Get normalized coordinates
        x_center, y_center, width, height = map(float, parts[1:5])
        
        # Convert normalized coordinates to absolute coordinates
        abs_x_center = int(x_center * w)
        abs_y_center = int(y_center * h)
        abs_width = int(width * w)
        abs_height = int(height * h)
        
        # Calculate bounding box corners
        x1 = max(0, abs_x_center - abs_width // 2)
        y1 = max(0, abs_y_center - abs_height // 2)
        x2 = min(w, x1 + abs_width)
        y2 = min(h, y1 + abs_height)
        
        # Extract the region
        symbol_crop = img[y1:y2, x1:x2]
        
        # Add padding if needed (10% on each side)
        padding = int(max(abs_width, abs_height) * 0.1)
        if padding > 0:
            symbol_crop_padded = cv2.copyMakeBorder(
                symbol_crop, 
                padding, padding, padding, padding, 
                cv2.BORDER_CONSTANT, 
                value=(255, 255, 255)  # White padding
            )
        else:
            symbol_crop_padded = symbol_crop
        
        # Create unique filename
        crop_filename = f"{cls_name}_{timestamp}_{idx}.png"
        crop_path = crops_dir / crop_filename
        
        # Save the crop
        cv2.imwrite(str(crop_path), symbol_crop_padded)
        crop_paths.append(crop_path)
        
        print(f"  Saved {cls_name} crop to: {crop_path}")
    
    return crop_paths

def copy_to_dataset(img_path, img):
    """Copy the image and its annotations to the dataset directories and extract symbol crops"""
    global current_annotations
    
    if not current_annotations:
        print("No annotations to copy. Please annotate the image first.")
        return
    
    # Create filenames
    base_filename = img_path.stem
    image_filename = f"{base_filename}{img_path.suffix}"
    label_filename = f"{base_filename}.txt"
    
    # Show current file counts
    print("\nCurrent files before saving:")
    print(f"  Train images: {len(list(TRAIN_IMAGES_DIR.glob('*.png')) + list(TRAIN_IMAGES_DIR.glob('*.jpg')))} files")
    print(f"  Val images: {len(list(VAL_IMAGES_DIR.glob('*.png')) + list(VAL_IMAGES_DIR.glob('*.jpg')))} files")
    
    # Ask user about train/val split
    print("\nWhere do you want to save this annotated image?")
    print("1. Training set")
    print("2. Validation set")
    print("3. Don't copy to dataset yet")
    
    choice = input("Enter your choice (1/2/3): ").strip()
    
    if choice == "1":
        # Save to training set
        img_save_path = TRAIN_IMAGES_DIR / image_filename
        label_save_path = TRAIN_LABELS_DIR / label_filename
        
        # Copy image and save annotations
        print(f"\nSaving to training set:")
        print(f"  Copying full image to: {img_save_path}")
        shutil.copy(img_path, img_save_path)
        
        print(f"  Saving labels ({len(current_annotations)} annotations) to: {label_save_path}")
        with open(label_save_path, "w") as f:
            for anno in current_annotations:
                f.write(f"{anno}\n")
        
        # Extract individual symbol crops
        print(f"\nExtracting individual symbol crops to: {TRAIN_CROPS_DIR}")
        crop_paths = extract_symbol_crops(img, current_annotations, "train")
                
        if img_save_path.exists() and label_save_path.exists():
            print("\n✅ Files saved successfully to training set!")
        else:
            print("\n❌ Some files failed to save:")
            if not img_save_path.exists(): print(f"  - Failed to save {img_save_path}")
            if not label_save_path.exists(): print(f"  - Failed to save {label_save_path}")
        
    elif choice == "2":
        # Save to validation set
        img_save_path = VAL_IMAGES_DIR / image_filename
        label_save_path = VAL_LABELS_DIR / label_filename
        
        # Copy image and save annotations
        print(f"\nSaving to validation set:")
        print(f"  Copying full image to: {img_save_path}")
        shutil.copy(img_path, img_save_path)
        
        print(f"  Saving labels ({len(current_annotations)} annotations) to: {label_save_path}")
        with open(label_save_path, "w") as f:
            for anno in current_annotations:
                f.write(f"{anno}\n")
        
        # Extract individual symbol crops
        print(f"\nExtracting individual symbol crops to: {VAL_CROPS_DIR}")
        crop_paths = extract_symbol_crops(img, current_annotations, "val")
                
        if img_save_path.exists() and label_save_path.exists():
            print("\n✅ Files saved successfully to validation set!")
        else:
            print("\n❌ Some files failed to save:")
            if not img_save_path.exists(): print(f"  - Failed to save {img_save_path}")
            if not label_save_path.exists(): print(f"  - Failed to save {label_save_path}")
            
    else:
        print("No copy made to dataset directories.")
        return
    
    # Updated file counts
    print("\nUpdated file counts:")
    print(f"  Train images: {len(list(TRAIN_IMAGES_DIR.glob('*.png')) + list(TRAIN_IMAGES_DIR.glob('*.jpg')))} files")
    print(f"  Train labels: {len(list(TRAIN_LABELS_DIR.glob('*.txt')))} files")
    print(f"  Train symbol crops: {len(list(TRAIN_CROPS_DIR.glob('*.png')))} files")
    print(f"  Val images: {len(list(VAL_IMAGES_DIR.glob('*.png')) + list(VAL_IMAGES_DIR.glob('*.jpg')))} files")
    print(f"  Val labels: {len(list(VAL_LABELS_DIR.glob('*.txt')))} files")
    print(f"  Val symbol crops: {len(list(VAL_CROPS_DIR.glob('*.png')))} files")
    
    # Ask if user wants to create data.yaml
    update_yaml = input("\nUpdate data.yaml file with these directories? (y/n): ").lower()
    if update_yaml == 'y':
        create_data_yaml()

def create_data_yaml():
    """Create the data.yaml file needed for YOLOv8 training"""
    yaml_content = f"""train: {TRAIN_IMAGES_DIR.resolve()}
val: {VAL_IMAGES_DIR.resolve()}

# number of classes
nc: {len(classes)}

# class names
names: {classes}
"""
    
    yaml_path = YAML_PATH # Use the globally defined absolute path
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    print(f"Created {yaml_path} for training")

def main():
    print("=" * 70)
    print("ENHANCED BLUEPRINT ANNOTATION TOOL")
    print("=" * 70)
    print("\nHOW ANNOTATIONS ARE SAVED:")
    print("1. Each image gets a YOLO format label file with the same name (.txt)")
    print("2. Each annotated symbol is ALSO extracted as an individual crop image")
    print("3. The original image is saved unmodified to train/images/ or val/images/")
    print("4. The annotations are saved to train/labels/ or val/labels/")
    print("5. Individual crops are saved to data/symbol_crops/train/ or .../val/")
    print("\nANNOTATION WORKFLOW:")
    print("1. Press 0, 1, or 2 to select class (evse, panel, gfi)")
    print("2. LEFT CLICK and DRAG to draw a rectangle around the symbol")
    print("3. Press 'c' to confirm or 'r' to retry the selection")
    print("4. Repeat steps 1-3 for all symbols in the image")
    print("5. Press 's' to save annotations and extract crops")
    print("6. Press 'n' to move to the next image")
    print("7. Press 'p' to toggle preview of annotations")
    print("=" * 70)
    
    input("Press Enter to start annotation...")
    
    # Run the annotation tool
    annotate_blueprints_interactive()
    
    print("\n=== Annotation Complete ===")
    print("Your dataset is now ready for training!")
    print(f"Full blueprint images: {TRAIN_IMAGES_DIR} and {VAL_IMAGES_DIR}")
    print(f"Label files: {TRAIN_LABELS_DIR} and {VAL_LABELS_DIR}")
    print(f"Individual symbol crops: {TRAIN_CROPS_DIR} and {VAL_CROPS_DIR}")
    print("\nNext steps:")
    print("1. Train the model with: python train/train_optimized.py")
    print("2. Use detect.py with your trained model")

if __name__ == "__main__":
    main() 