#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
from tqdm import tqdm

# Add app directory to sys.path to find the detector module
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'app'))
sys.path.insert(0, str(project_root))

# Standard model path - using tiny_symbol_detector4 model
STANDARD_MODEL_PATH = str(project_root / "runs" / "train" / "tiny_symbol_detector4" / "weights" / "best.pt")

try:
    # Import the detector class and NMS function
    from detector import BlueprintDetector, apply_nms
except ImportError as e:
    print(f"Error importing detector module: {e}")
    print("Ensure app/detector.py exists and necessary dependencies (ultralytics, etc.) are installed.")
    sys.exit(1)


def standardize_boxes(detections: List[List[float]], max_aspect_ratio: float = 3.0) -> List[List[float]]:
    """
    Standardize bounding boxes to prevent overly wide boxes.
    
    Args:
        detections: List of [x1, y1, x2, y2, conf, class_id] lists
        max_aspect_ratio: Maximum allowed aspect ratio (width/height)
        
    Returns:
        List of standardized boxes
    """
    standardized = []
    for box in detections:
        x1, y1, x2, y2, conf, class_id = box
        width, height = x2 - x1, y2 - y1
        
        # Skip invalid boxes
        if width <= 0 or height <= 0:
            continue

        aspect = width / height
        
        # If box is too wide, adjust it
        if aspect > max_aspect_ratio:
            center_x = (x1 + x2) / 2
            new_width = height * max_aspect_ratio
            new_x1 = max(0, center_x - new_width / 2)
            new_x2 = center_x + new_width / 2
            standardized.append([new_x1, y1, new_x2, y2, conf, class_id])
        else:
            standardized.append(box)
            
    return standardized


def process_file_multi_scale(
    detector: BlueprintDetector, 
    image_path: str, 
    output_dir: str = "app/results", 
    tile_sizes: List[int] = [640],
    overlaps: List[float] = [0.4],
    nms_threshold: float = 0.6,
    final_nms_threshold: float = 0.4,
    use_flips: bool = False,
    save_tiles: bool = False,
    max_aspect_ratio: float = 3.0
) -> Tuple[Optional[str], List]:
    """
    Process an image using multi-scale tiled inference with ensemble techniques.
    
    Args:
        detector: The detector model
        image_path: Path to the input image
        output_dir: Directory to save output
        tile_sizes: List of tile sizes to use
        overlaps: List of overlap ratios to use for each tile size
        nms_threshold: NMS threshold for each scale
        final_nms_threshold: NMS threshold for merging detections across scales
        use_flips: Whether to apply horizontal/vertical flips for test-time augmentation
        save_tiles: Whether to save individual tiles for debugging
        max_aspect_ratio: Maximum allowed aspect ratio for bounding boxes
        
    Returns:
        Tuple of (output_image_path, detections)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # If save_tiles is True, create a subdirectory for tiles
    tiles_dir = None
    if save_tiles:
        tiles_dir = output_path / "tiles" / Path(image_path).stem
        tiles_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original image for verification
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image {image_path}")
        return None, []
    
    h, w = original_image.shape[:2]
    print(f"Image dimensions: {w}x{h}")
    
    # If image is very small, resize it to have a minimum dimension
    min_dim = 640
    if min(h, w) < min_dim:
        scale_factor = min_dim / min(h, w)
        print(f"Image is too small. Resizing by factor {scale_factor:.2f}")
        original_image = cv2.resize(original_image, (
            int(w * scale_factor), int(h * scale_factor)
        ))
        h, w = original_image.shape[:2]
        print(f"New dimensions: {w}x{h}")
    
    # For large images, add a larger tile size if not already specified
    if max(h, w) > 2000 and 1280 not in tile_sizes:
        tile_sizes.append(1280)
        overlaps.append(0.6)
    
    # Initialize empty list for all detections across scales
    all_detections = []
    scale_count = len(tile_sizes)
    
    # Process each tile size with corresponding overlap
    print(f"Running detection with {scale_count} scale{'s' if scale_count > 1 else ''}")
    for i, (tile_size, overlap) in enumerate(zip(tile_sizes, overlaps)):
        print(f"Scale {i+1}/{scale_count}: Tile size {tile_size}px, overlap {overlap:.1f}")
        
        # Run detection for this scale
        try:
            start_time = time.time()
            
            # Original orientation
            detections = detector.detect_tiled(
                image_path,
                tile_size=tile_size,
                overlap=overlap,
                nms_threshold=nms_threshold,  # Using the higher NMS threshold
                save_tiles=save_tiles,
                tiles_dir=tiles_dir,
                tile_prefix=f"scale{i+1}_orig_"
            )
            all_detections.extend(detections)
            
            # Test-time augmentation with flips if enabled
            if use_flips:
                # Load image and create flipped versions
                img = cv2.imread(image_path)
                if img is not None:
                    # Horizontal flip
                    img_h_flip = cv2.flip(img, 1)  # 1 for horizontal flip
                    h_flip_path = str(output_path / f"temp_h_flip_{Path(image_path).stem}.jpg")
                    cv2.imwrite(h_flip_path, img_h_flip)
                    
                    # Detect on horizontally flipped image
                    flip_h_detections = detector.detect_tiled(
                        h_flip_path,
                        tile_size=tile_size,
                        overlap=overlap,
                        nms_threshold=nms_threshold,  # Using the higher NMS threshold
                        save_tiles=save_tiles,
                        tiles_dir=tiles_dir,
                        tile_prefix=f"scale{i+1}_hflip_"
                    )
                    
                    # Transform coordinates back to original image space
                    for det in flip_h_detections:
                        # Flip x-coordinate: new_x = image_width - old_x
                        # bbox format is [x1, y1, x2, y2, confidence, class]
                        img_width = img.shape[1]
                        det[0] = img_width - det[2]  # x1 becomes width - x2
                        det[2] = img_width - det[0]  # x2 becomes width - x1
                    
                    all_detections.extend(flip_h_detections)
                    
                    # Clean up temporary files
                    os.remove(h_flip_path)
            
            end_time = time.time()
            print(f"  - Scale {i+1} detection took {end_time - start_time:.2f} seconds")
            print(f"  - Found {len(detections)} potential symbols at this scale")
            
        except Exception as e:
            print(f"Error during detection at scale {i+1}: {e}")
            continue  # Try next scale even if this one fails
    
    # If no detections at any scale, return early
    if not all_detections:
        print("No detections found at any scale.")
        return None, []
    
    # Apply final NMS across all scales to merge detections
    # Using the higher final_nms_threshold to reduce duplicates
    print(f"Applying final NMS with threshold {final_nms_threshold} across {len(all_detections)} raw detections")
    
    # First standardize boxes to fix overly wide ones
    standardized_detections = standardize_boxes(all_detections, max_aspect_ratio=max_aspect_ratio)
    # Then apply NMS to remove duplicates
    final_detections = apply_nms(standardized_detections, final_nms_threshold)
    print(f"Final detection count after cross-scale NMS: {len(final_detections)}")

    # Draw detections on the original image
    print("Drawing detections...")
    annotated_image = detector.draw_detections(original_image, final_detections)

    # Save annotated image
    output_file_path_str = None
    if annotated_image is not None:
        basename = Path(image_path).stem
        output_filename = f"{basename}_multi_scale_detected.png"
        output_file = output_path / output_filename
        try:
            is_success = cv2.imwrite(str(output_file), annotated_image)
            if is_success:
                print(f"Saved annotated image to {output_file}")
                output_file_path_str = str(output_file)
            else:
                print(f"Warning: cv2.imwrite failed to save {output_file}")
        except Exception as e:
            print(f"Error saving annotated image to {output_file}: {e}")
    
    return output_file_path_str, final_detections


def process_directory(
    detector: BlueprintDetector, 
    input_dir: str, 
    output_dir: str = "app/results", 
    tile_sizes: List[int] = [640],
    overlaps: List[float] = [0.4],
    nms_threshold: float = 0.6,
    final_nms_threshold: float = 0.4,
    use_flips: bool = False,
    save_tiles: bool = False,
    max_aspect_ratio: float = 3.0
) -> List[Tuple[str, List]]:
    """Process all images in a directory using multi-scale tiled inference"""
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Input directory {input_dir} does not exist.")
        return []

    image_files = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg"))
    if not image_files:
        print(f"No image files found in {input_dir}")
        return []

    print(f"Found {len(image_files)} images to process in {input_dir}")
    results = []
    for image_file in image_files:
        print("-" * 50)
        print(f"Processing: {image_file.name}")
        output_file, detections = process_file_multi_scale(
            detector,
            str(image_file),
            output_dir,
            tile_sizes=tile_sizes,
            overlaps=overlaps,
            nms_threshold=nms_threshold,
            final_nms_threshold=final_nms_threshold,
            use_flips=use_flips,
            save_tiles=save_tiles,
            max_aspect_ratio=max_aspect_ratio
        )
        if output_file or detections:
            results.append((output_file, detections))

    return results


def display_results(results: List[Tuple[str, List]], max_display_height: int = 1000):
    """Display annotated images in separate windows."""
    if not results:
        print("No results to display.")
        return
        
    print("Displaying results. Press any key to close each window.")
    windows_created = False
    for output_file, _ in results:
        if output_file and Path(output_file).exists():
            try:
                image = cv2.imread(output_file)
                if image is not None:
                    h, w = image.shape[:2]
                    if h > max_display_height:
                        scale = max_display_height / h
                        display_img = cv2.resize(image, (int(w * scale), max_display_height))
                    else:
                        display_img = image
                    cv2.imshow(f"Detection - {Path(output_file).name}", display_img)
                    windows_created = True
                    cv2.waitKey(0)
                else:
                    print(f"Could not read result image for display: {output_file}")
            except Exception as e:
                print(f"Error displaying image {output_file}: {e}")
        elif output_file:
            print(f"Result image file not found, cannot display: {output_file}")
             
    if windows_created:
        cv2.destroyAllWindows()
    else:
        print("No valid images were found to display.")


def main():
    parser = argparse.ArgumentParser(description="Detect electrical symbols in blueprint images using tiled inference.")
    parser.add_argument("--input", "-i", default="data/images", help="Input image file or directory (defaults to data/images)")
    parser.add_argument("--output", "-o", default="app/results", help="Output directory for annotated images")
    parser.add_argument("--model", "-m", default=STANDARD_MODEL_PATH, help=f"Path to model file (defaults to {STANDARD_MODEL_PATH})")
    parser.add_argument("--conf", "-c", type=float, default=0.65, help="Confidence threshold (0-1). Higher values reduce false positives.")
    parser.add_argument("--scales", "-s", type=str, default="640", help="Comma-separated list of tile sizes for detection. Single scale (640) is faster, multiple scales may improve accuracy.")
    parser.add_argument("--overlaps", "-ov", type=str, default="0.4", help="Comma-separated list of overlap fractions for each scale. Higher values may improve detection of symbols at tile boundaries but increase processing time.")
    parser.add_argument("--nms", "-n", type=float, default=0.6, help="NMS threshold for each scale. Higher values (0.5-0.7) reduce duplicate detections within each scale.")
    parser.add_argument("--final-nms", "-fn", type=float, default=0.4, help="NMS threshold for merging across scales. Higher values (0.4-0.6) reduce duplicate detections in final output.")
    parser.add_argument("--use-flips", action="store_true", help="Enable test-time augmentation with flips. Improves accuracy but doubles processing time.")
    parser.add_argument("--save-tiles", action="store_true", help="Save individual tiles for debugging.")
    parser.add_argument("--view", "-v", action="store_true", help="Display annotated images in window(s).")
    parser.add_argument("--max-aspect", type=float, default=3.0, 
                       help="Maximum aspect ratio (width/height) for bounding boxes. Helps prevent excessively wide boxes.")

    args = parser.parse_args()

    # Process scale and overlap arguments
    try:
        tile_sizes = [int(s) for s in args.scales.split(",")]
        overlaps = [float(o) for o in args.overlaps.split(",")]
        
        # Validate
        if len(tile_sizes) != len(overlaps):
            print("Error: Number of tile sizes must match number of overlap values")
            sys.exit(1)
            
        for o in overlaps:
            if not (0 <= o < 1):
                print("Error: Overlap must be between 0.0 and 0.9")
                sys.exit(1)
    except ValueError as e:
        print(f"Error parsing scales or overlaps: {e}")
        sys.exit(1)

    # Validate other arguments
    if not (0 <= args.nms <= 1) or not (0 <= args.final_nms <= 1):
        print("Error: NMS thresholds must be between 0.0 and 1.0")
        sys.exit(1)
    if not (0 < args.conf <= 1):
        print("Error: Confidence threshold must be between 0.0 (exclusive) and 1.0")
        sys.exit(1)

    # Initialize detector
    try:
        detector = BlueprintDetector(model_path=args.model, conf_threshold=args.conf)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Failed to initialize detector: {e}")
        sys.exit(1)

    # Process input
    input_path = Path(args.input)
    all_results = [] 

    if input_path.is_file():
        output_file, detections = process_file_multi_scale(
            detector,
            str(input_path),
            args.output,
            tile_sizes=tile_sizes,
            overlaps=overlaps,
            nms_threshold=args.nms,
            final_nms_threshold=args.final_nms,
            use_flips=args.use_flips,
            save_tiles=args.save_tiles,
            max_aspect_ratio=args.max_aspect
        )
        if output_file or detections:
            all_results.append((output_file, detections))

    elif input_path.is_dir():
        all_results = process_directory(
            detector,
            str(input_path),
            args.output,
            tile_sizes=tile_sizes,
            overlaps=overlaps,
            nms_threshold=args.nms,
            final_nms_threshold=args.final_nms,
            use_flips=args.use_flips,
            save_tiles=args.save_tiles,
            max_aspect_ratio=args.max_aspect
        )
    else:
        print(f"Error: Input path {args.input} is neither a file nor a directory.")
        sys.exit(1)
        
    # Print summary of detections
    total_detected_count = sum(len(dets) for _, dets in all_results)
    print(f"\n--- Processing Complete --- ")
    print(f"Processed {len(all_results)} image(s). Found {total_detected_count} symbols in total.")
    if all_results:
        print("Summary per image:")
        for i, (out_file, dets) in enumerate(all_results):
            filename = Path(out_file).name if out_file else f"Image {i+1} (save/annotation failed)"
            print(f"  - {filename}: {len(dets)} symbols detected.")

    if args.view:
        display_results(all_results)


if __name__ == "__main__":
    main()
