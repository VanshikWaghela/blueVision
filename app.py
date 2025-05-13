#!/usr/bin/env python3
import os
import sys
import tempfile
import time
import cv2
import numpy as np
import gradio as gr
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

# Importing the detector
from app.detector import BlueprintDetector, apply_nms

# we will use my custom trained model from app/models/weights
MODEL_PATH = str(project_root / "app" / "models" / "weights" / "best.pt")

# Ensure results directory exists
RESULTS_DIR = project_root / "app" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set page title and description
title = "Blueprint Electrical Symbol Detector"
description = """
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
  <div>
    <h1 style="font-weight: 900; font-size: 3rem;">Blueprint Symbol Detector</h1>
    <p style="margin-bottom: 10px; font-size: 94%">
      Detect electrical symbols (EVSE, Panel, GFI) in blueprint images using custom trained YOLOv8n
    </p>
  </div>
</div>
"""

# Helper function to load the detector with custom settings
def load_detector(confidence: float) -> BlueprintDetector:
    """Load the detector with the specified confidence threshold"""
    print(f"Loading detector with confidence threshold: {confidence}")
    detector = BlueprintDetector(
        model_path=MODEL_PATH,
        conf_threshold=confidence
    )
    return detector

# Function to process blueprint with custom settings
def process_blueprint(
    input_image: np.ndarray,
    confidence: float = 0.65,
    detection_mode: str = "Fast",
    show_boxes: bool = True
) -> Tuple[Dict, np.ndarray]:
    """
    Process a blueprint image with the detector.

    Args:
        input_image: The blueprint image as a numpy array
        confidence: Confidence threshold (0-1)
        detection_mode: "Fast" or "Accurate" detection
        show_boxes: Whether to draw bounding boxes

    Returns:
        Tuple of (results_dict, annotated_image)
    """
    # Ensure we have a valid image
    if input_image is None:
        raise ValueError("No image provided")

    # Save input image to a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_img_path = Path(temp_dir) / "input_image.png"
    cv2.imwrite(str(temp_img_path), input_image)

    start_time = time.time()

    try:
        # Load detector with specified confidence
        detector = load_detector(confidence)

        # Process based on mode
        if detection_mode == "Accurate":
            print("Using ACCURATE mode with multi-scale detection")
            # Determine scales based on image size
            h, w = input_image.shape[:2]
            tiles = [640, 960, 1280] if max(h, w) > 1000 else [640, 960]
            overlaps = [0.5] * len(tiles)

            # Process all scales
            all_detections = []
            for i, (tile_size, overlap) in enumerate(zip(tiles, overlaps)):
                print(f"Processing scale {i+1}/{len(tiles)}: tile size {tile_size}px, overlap {overlap:.1f}")

                # Original orientation
                detections = detector.detect_tiled(
                    image_path=str(temp_img_path),
                    tile_size=tile_size,
                    overlap=overlap,
                    nms_threshold=0.6,
                )
                all_detections.extend(detections)

                # Horizontal flip for test-time augmentation
                img = cv2.imread(str(temp_img_path))
                img_h_flip = cv2.flip(img, 1)
                flip_path = str(temp_img_path) + "_flip.jpg"
                cv2.imwrite(flip_path, img_h_flip)

                flip_detections = detector.detect_tiled(
                    image_path=flip_path,
                    tile_size=tile_size,
                    overlap=overlap,
                    nms_threshold=0.6,
                )

                # Transform coordinates back
                img_width = img.shape[1]
                for det in flip_detections:
                    det[0] = img_width - det[2]
                    det[2] = img_width - det[0]

                all_detections.extend(flip_detections)
                os.remove(flip_path)

            # Standardize detections and apply NMS
            max_aspect_ratio = 3.0
            standardized_detections = []
            for box in all_detections:
                x1, y1, x2, y2, conf, class_id = box
                width, height = x2 - x1, y2 - y1

                # Skip invalid boxes
                if width <= 0 or height <= 0:
                    continue

                aspect = width / height
                if aspect > max_aspect_ratio:
                    center_x = (x1 + x2) / 2
                    new_width = height * max_aspect_ratio
                    new_x1 = max(0, center_x - new_width / 2)
                    new_x2 = center_x + new_width / 2
                    standardized_detections.append([new_x1, y1, new_x2, y2, conf, class_id])
                else:
                    standardized_detections.append(box)

            detections = apply_nms(standardized_detections, 0.4)

        else:  # Fast mode
            print("Using FAST mode with single-scale detection")
            detections = detector.detect_tiled(
                image_path=str(temp_img_path),
                tile_size=640,
                overlap=0.3,
                nms_threshold=0.45,
            )

        # Create annotated image if requested
        if show_boxes:
            annotated_image = detector.draw_detections(input_image, detections)
        else:
            annotated_image = input_image.copy()

        # Format results
        formatted_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            class_id = int(class_id)
            class_name = detector.class_names.get(class_id, f"Class {class_id}")

            formatted_detections.append({
                "label": class_name,
                "confidence": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })

        processing_time = time.time() - start_time

        # Create results dictionary
        results = {
            "detections": formatted_detections,
            "processing_time": round(processing_time, 2),
            "detection_mode": detection_mode,
            "confidence_threshold": confidence,
            "symbol_count": {
                "total": len(formatted_detections),
                "evse": sum(1 for d in formatted_detections if d["label"].lower() == "evse"),
                "panel": sum(1 for d in formatted_detections if d["label"].lower() == "panel"),
                "gfi": sum(1 for d in formatted_detections if d["label"].lower() == "gfi"),
            }
        }

        return results, annotated_image

    except Exception as e:
        print(f"Error processing image: {e}")
        raise
    finally:
        # Clean up temp files
        if os.path.exists(str(temp_img_path)):
            os.remove(str(temp_img_path))
        try:
            os.rmdir(temp_dir)
        except:
            pass

# Function to format results as HTML
def format_results_html(results: Dict) -> str:
    """Format detection results as HTML for display"""
    if not results:
        return "<div>No results available</div>"

    html = f"""
    <div style="font-family: sans-serif; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">
        <h3 style="margin-top: 0;">Detection Results</h3>
        <p>Processing time: {results['processing_time']} seconds | Mode: {results['detection_mode']}</p>
        <p>Confidence threshold: {results['confidence_threshold']}</p>

        <h4>Symbol Count:</h4>
        <ul>
            <li>Total: {results['symbol_count']['total']}</li>
            <li>EVSE: {results['symbol_count']['evse']}</li>
            <li>Panel: {results['symbol_count']['panel']}</li>
            <li>GFI: {results['symbol_count']['gfi']}</li>
        </ul>

        <h4>Detailed Detections:</h4>
        <div style="max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px;">
    """

    for i, det in enumerate(results["detections"]):
        html += f"""
        <div style="margin-bottom: 5px;">
            <strong>{i+1}. {det['label']}</strong> (Confidence: {det['confidence']:.2f})
            <br>
            <small>Box: [{int(det['bbox'][0])}, {int(det['bbox'][1])}, {int(det['bbox'][2])}, {int(det['bbox'][3])}]</small>
        </div>
        """

    html += """
        </div>
    </div>
    """

    return html

# Define the Gradio interface
def create_interface():
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.HTML(description)

        with gr.Row():
            with gr.Column(scale=2):
                input_image = gr.Image(type="numpy", label="Upload Blueprint")

                with gr.Row():
                    with gr.Column(scale=1):
                        confidence = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.65,
                            step=0.05,
                            label="Confidence Threshold"
                        )

                    with gr.Column(scale=1):
                        detection_mode = gr.Radio(
                            ["Fast", "Accurate"],
                            value="Fast",
                            label="Detection Mode",
                            info="Fast: ~10-30s | Accurate: ~1-3 min"
                        )

                show_boxes = gr.Checkbox(value=True, label="Show Detection Boxes")

                submit_btn = gr.Button("Detect Symbols", variant="primary")

            with gr.Column(scale=2):
                results_html = gr.HTML(label="Results")
                output_image = gr.Image(type="numpy", label="Annotated Blueprint")

        # Set up the submission action
        submit_btn.click(
            fn=process_blueprint,
            inputs=[input_image, confidence, detection_mode, show_boxes],
            outputs=[results_html, output_image],
            api_name="detect"
        )

        # Examples section removed to protect company data
        # If you have sample blueprint images, you can add them as examples:
        # gr.Examples(
        #     examples=[
        #         ["path/to/example1.png", 0.65, "Fast", True],
        #         ["path/to/example2.png", 0.5, "Accurate", True],
        #     ],
        #     inputs=[input_image, confidence, detection_mode, show_boxes],
        # )

        # Add documentation
        gr.HTML("""
        <div style="text-align: left; max-width: 650px; margin: 20px auto;">
            <h3>How to Use</h3>
            <ol>
                <li><strong>Upload</strong> a blueprint image (PNG, JPG)</li>
                <li>Adjust <strong>confidence threshold</strong> (higher = fewer detections but more accurate)</li>
                <li>Choose <strong>detection mode</strong>:
                    <ul>
                        <li><strong>Fast</strong>: Single-scale detection (faster but may miss small symbols)</li>
                        <li><strong>Accurate</strong>: Multi-scale detection with flips (more accurate but slower)</li>
                    </ul>
                </li>
                <li>Toggle <strong>show detection boxes</strong> to visualize detections</li>
                <li>Click <strong>Detect Symbols</strong> to process the image</li>
            </ol>

            <h3>Troubleshooting</h3>
            <ul>
                <li>If symbols are missed, try lowering the confidence threshold</li>
                <li>For better accuracy, use the "Accurate" mode</li>
                <li>Processing large blueprints may take longer</li>
            </ul>
        </div>
        """)

    return demo

# Create and launch the interface
demo = create_interface()

# For Hugging Face Spaces
if __name__ == "__main__":
    demo.launch()
