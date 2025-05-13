#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
from tqdm import tqdm # Add tqdm for progress bar

# Add project root to sys.path to allow importing ultralytics if needed
# This might be necessary depending on how the environment is set up
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Ultralytics not installed. Please install with: pip install ultralytics")
    # In a server context, raising an exception might be better than sys.exit
    raise RuntimeError("Ultralytics not installed.")

# Define colors for each class (BGR format) - Keep if needed for drawing, maybe remove later
COLORS = {
    0: (0, 0, 255),    # EVSE - Red
    1: (0, 255, 0),    # Panel - Green
    2: (255, 0, 0),    # GFI - Blue
}

# Standard model path - using tiny_symbol_detector4 model
STANDARD_MODEL_PATH = str(project_root / "runs" / "train" / "tiny_symbol_detector4" / "weights" / "best.pt")
# Backup model path in app/models/weights
APP_MODEL_PATH = str(project_root / "app" / "models" / "weights" / "best.pt")

def apply_nms(boxes: List[List[float]], iou_threshold: float = 0.45) -> List[List[float]]:
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping detections.
    
    Args:
        boxes: List of [x1, y1, x2, y2, conf, class_id] lists 
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of boxes after NMS
    """
    if not boxes:
        return []
        
    # Convert list of boxes to numpy array
    boxes_array = np.array(boxes)
    
    # Get coordinates, scores, and class IDs
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    scores = boxes_array[:, 4]
    class_ids = boxes_array[:, 5]
    
    # Calculate areas of all boxes
    areas = (x2 - x1) * (y2 - y1)
    
    # Order by confidence score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        # Pick the box with highest confidence
        i = order[0]
        keep.append(i)
        
        # Find IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        
        # IoU = intersection / (area1 + area2 - intersection)
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)
        
        # Add class_id check - only apply NMS within same class
        class_matches = (class_ids[i] == class_ids[order[1:]])
        
        # Apply stricter threshold for same class overlaps
        to_remove = np.where((iou > iou_threshold) & class_matches)[0] + 1
        
        # Update order by removing suppressed boxes
        order = np.delete(order, to_remove)
        order = np.delete(order, 0)
    
    return [boxes[i] for i in keep]


class BlueprintDetector:
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.65):
        """Initialize the detector with a model path"""
        self.conf_threshold = conf_threshold
        self.model = self._load_model(model_path)
        self.model_path = model_path  # Store for reference
        self.class_names = self._get_class_names()
        print(f"Detector initialized with confidence threshold: {conf_threshold}")
        print(f"Classes: {self.class_names}")

    def _load_model(self, model_path: Optional[str]) -> YOLO:
        """
        Load a YOLOv8 model from file or search common paths.
        
        Args:
            model_path: Optional path to model file
            
        Returns:
            Loaded YOLO model
        """
        # List of common paths to search for models, in priority order
        common_paths = [
            # Use provided model path first
            model_path if model_path else None,
            # Standard paths in priority order
            STANDARD_MODEL_PATH,  # Primary model path in runs/train  
            APP_MODEL_PATH,       # App model directory
            "runs/train/tiny_symbol_detector4/weights/best.pt",  # For backward compatibility
            "app/models/weights/best.pt",                        # For backward compatibility
            # Fall back to default YOLOv8 model
            "yolov8n.pt"
        ]
        
        # Try each path in order
        for path in common_paths:
            if path and Path(path).exists():
                print(f"Loading model from path: {path}")
                return YOLO(path)
        
        # If we get here, no model was found
        raise FileNotFoundError(f"No model file found. Provide a valid model path or place model in one of the common paths: {', '.join(str(p) for p in common_paths if p)}")

    def _get_class_names(self) -> List[str]:
        """
        Get class names from the model.
        
        Returns:
            List of class names
        """
        try:
            return self.model.names
        except (AttributeError, KeyError):
            # Default class names if model doesn't have them
            return ['evse', 'panel', 'gfi']

    def detect(self, image_path: str) -> List[List[float]]:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of [x1, y1, x2, y2, conf, class_id] lists for each detection
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Run inference with YOLOv8
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False,
            save=False
        )
        
        detections = []
        result = results[0]  # First (and only) image result
        
        # Extract boxes, confidences, and class IDs
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = result.boxes.conf.cpu().numpy()  # confidence
            class_ids = result.boxes.cls.cpu().numpy()  # class IDs
            
            # Combine into detection format: [x1, y1, x2, y2, conf, class_id]
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                detections.append([float(x1), float(y1), float(x2), float(y2), float(confs[i]), float(class_ids[i])])
        
        return detections

    def detect_tiled(
        self, 
        image_path: str, 
        tile_size: int = 640, 
        overlap: float = 0.5, 
        nms_threshold: float = 0.5,
        save_tiles: bool = False,
        tiles_dir: Optional[str] = None,
        tile_prefix: str = ""
    ) -> List[List[float]]:
        """
        Detect objects in an image using tiled inference.
        
        Args:
            image_path: Path to image file
            tile_size: Size of tiles for inference
            overlap: Overlap ratio between adjacent tiles
            nms_threshold: IoU threshold for NMS
            save_tiles: Flag to save tiles for debugging
            tiles_dir: Directory to save tiles
            tile_prefix: Prefix for saved tile filenames
            
        Returns:
            List of [x1, y1, x2, y2, conf, class_id] lists for each detection
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        h, w = img.shape[:2]
        
        # Calculate tile parameters
        stride = int(tile_size * (1 - overlap))
        
        # Calculate the number of tiles in x and y directions
        nx = max(1, int(np.ceil((w - tile_size) / stride)) + 1)
        ny = max(1, int(np.ceil((h - tile_size) / stride)) + 1)
        
        # Initialize list to hold all detections
        all_detections = []
        
        # Debug print
        if nx * ny > 1:
            print(f"Processing image in {nx}x{ny}={nx*ny} tiles (size={tile_size}, overlap={overlap:.2f})")
        
        # Process each tile
        tile_count = 0
        for y in range(0, max(0, h - tile_size) + 1, stride):
            for x in range(0, max(0, w - tile_size) + 1, stride):
                # Ensure the tile doesn't go out of bounds
                x1 = min(x, w - tile_size)
                y1 = min(y, h - tile_size)
                x2 = x1 + tile_size
                y2 = y1 + tile_size
                
                # Extract tile
                tile = img[y1:y2, x1:x2]
                
                # Save tile for debugging if requested
                if save_tiles and tiles_dir:
                    tile_filename = f"{tile_prefix}tile_{tile_count:03d}_x{x1}_y{y1}.jpg"
                    tile_path = Path(tiles_dir) / tile_filename
                    cv2.imwrite(str(tile_path), tile)
                
                # Create a temporary file for this tile
                tile_filename = f"temp_tile_{tile_count}.jpg"
                temp_tile_path = Path(image_path).parent / tile_filename
                cv2.imwrite(str(temp_tile_path), tile)
                
                try:
                    # Run inference on tile
                    tile_detections = self.detect(str(temp_tile_path))
                    
                    # Adjust coordinates to be relative to the original image
                    for det in tile_detections:
                        det[0] += x1  # Adjust x1
                        det[1] += y1  # Adjust y1
                        det[2] += x1  # Adjust x2
                        det[3] += y1  # Adjust y2
                    
                    # Add to all detections
                    all_detections.extend(tile_detections)
                    
                finally:
                    # Clean up temporary file
                    if temp_tile_path.exists():
                        os.remove(str(temp_tile_path))
                
                tile_count += 1
        
        # Apply NMS to remove duplicate detections
        if all_detections:
            all_detections = apply_nms(all_detections, nms_threshold)
        
        return all_detections

    def draw_detections(self, image: np.ndarray, detections: List[List[float]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: Image to draw on
            detections: List of [x1, y1, x2, y2, conf, class_id] lists
            
        Returns:
            Annotated image
        """
        if image is None:
            return None
        
        # Create a copy of the image to draw on
        annotated_img = image.copy()
        
        # Colors for different classes
        colors = COLORS
        
        # Font and text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        # Draw each detection
        for detection in detections:
            # Extract coordinates and other info
            try:
                x1, y1, x2, y2, conf, class_id = detection
                class_id = int(class_id)
                
                # Get class name
                class_name = self.class_names.get(class_id, f"Class {class_id}")
                
                # Get color for this class
                color = colors.get(class_id, (0, 255, 255))  # Default to yellow if class not found
                
                # Draw bounding box
                cv2.rectangle(annotated_img, 
                              (int(x1), int(y1)), 
                              (int(x2), int(y2)), 
                              color, 
                              thickness)
                
                # Prepare label text with confidence
                label = f"{class_name} {conf:.2f}"
                
                # Calculate text size
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                # Draw label background
                cv2.rectangle(annotated_img, 
                              (int(x1), int(y1) - text_height - 5), 
                              (int(x1) + text_width, int(y1)), 
                              color, 
                              -1)  # -1 means filled
                
                # Draw label text
                cv2.putText(annotated_img, 
                            label, 
                            (int(x1), int(y1) - 5), 
                            font, 
                            font_scale, 
                            (0, 0, 0),  # Black text
                            thickness)
            except Exception as e:
                print(f"Error drawing detection: {e}")
                continue
        
        return annotated_img

# Example of how to use if run directly (for testing)
if __name__ == "__main__":
    print("Testing BlueprintDetector class...")
    # Use a placeholder image path for testing
    test_image_path = "../data/images/sample_blueprint.png"
    test_image_path_obj = Path(test_image_path)

    if test_image_path_obj.exists():
        try:
            print("Initializing detector (will search for model)...")
            # Initialize with default confidence, let it find the model
            detector = BlueprintDetector(conf_threshold=0.65)
            
            print(f"Running detection on test image: {test_image_path}")
            detections = detector.detect_tiled(
                str(test_image_path_obj.resolve()),
                tile_size=640,
                overlap=0.3,
                nms_threshold=0.45
            )
            
            print(f"Detection complete. Found {len(detections)} symbols.")
            for i, det in enumerate(detections):
                print(f"  {i+1}. Class: {det['class_name']}, Confidence: {det['confidence']:.2f}, BBox: {det['bbox']}")
        
        except (FileNotFoundError, RuntimeError, ValueError) as e:
             print(f"Error during detector test: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during detector test: {e}")
             
    else:
        print(f"Test image not found at expected location: {test_image_path}")
        print("Skipping detector test.") 