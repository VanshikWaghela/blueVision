#!/usr/bin/env python3
import os
import sys
import shutil
import tempfile
import time
import base64
import cv2
import numpy as np
import uuid # For unique filenames
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Request
# Import StaticFiles for serving images and JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse # Keep HTMLResponse for root
from pydantic import BaseModel, Field

# Ensure the app directory is in sys.path to find detector
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

# Import the detector class
try:
    from detector import BlueprintDetector, apply_nms
except ImportError as e:
    print(f"FATAL ERROR: Could not import BlueprintDetector from detector.py: {e}")
    raise
except (FileNotFoundError, RuntimeError) as e:
     print(f"FATAL ERROR: Failed to initialize BlueprintDetector during import: {e}")
     raise

# --- Constants ---
STATIC_DIR_NAME = "static"
RESULTS_SUBDIR = "results"
STATIC_DIR_PATH = app_dir / STATIC_DIR_NAME
RESULTS_DIR_PATH = STATIC_DIR_PATH / RESULTS_SUBDIR
# Ensure results directory exists (redundant with terminal cmd but safe)
RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)

# Path to the standard model - using tiny_symbol_detector4 model
STANDARD_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "runs" / "train" / "tiny_symbol_detector4" / "weights" / "best.pt")

# --- Global Store for Model ---
ml_models: Dict[str, Any] = {}

# --- Lifespan Management (Load/Unload Model) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the YOLO model and setup static files."""
    print("--- Initializing Application --- ")
    app.mount(f"/{STATIC_DIR_NAME}", StaticFiles(directory=STATIC_DIR_PATH), name=STATIC_DIR_NAME)
    print(f"Mounted static directory at '/{STATIC_DIR_NAME}' serving from {STATIC_DIR_PATH}")

    # Use the standard model path defined above
    print(f"Attempting to load model: {STANDARD_MODEL_PATH}")
    try:
        # Initialize with confidence threshold of 0.65 as recommended
        ml_models["blueprint_detector"] = BlueprintDetector(
             model_path=STANDARD_MODEL_PATH,
             conf_threshold=0.65
        )
        print(f"--- BlueprintDetector Model Loaded Successfully (Conf Threshold: 0.65) --- ")
    except (FileNotFoundError, RuntimeError) as e:
         print(f"FATAL ERROR during model loading in lifespan: {e}")
         print(f"Please ensure the model file exists at: {STANDARD_MODEL_PATH}")
         raise
    except Exception as e:
         print(f"An unexpected error occurred during model loading: {e}")
         raise

    yield

    print("--- Shutting Down Application --- ")
    ml_models.clear()
    print("--- BlueprintDetector Model Unloaded --- ")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="blueVision ⚡️ API",
    description="Detects electrical symbols (EVSE, Panel, GFI) in blueprint images using YOLOv8. Returns JSON with detections and URL to annotated image.",
    version="1.4.0", # Incremented version
    lifespan=lifespan
)

# --- Pydantic Models for API Response ---
class DetectionResult(BaseModel):
    label: str = Field(..., description="Name of the detected class (e.g., 'evse', 'panel', 'gfi')")
    confidence: float = Field(..., description="Confidence score of the detection (0.0 to 1.0)")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x, y, width, height]")
    class Config:
        json_schema_extra = {
            "example": {
                "label": "evse",
                "confidence": 0.87,
                "bbox": [150, 200, 35, 30]
            }
        }
class ApiResponse(BaseModel):
    detections: List[DetectionResult] = Field(..., description="List of detected symbols.")
    image_url: Optional[str] = Field(None, description="URL to the annotated image saved on the server, if generated.")
    processing_time: float = Field(..., description="Time taken to process the image in seconds")
    mode: str = Field(..., description="Detection mode used (fast or accurate)")
    class Config:
         json_schema_extra = {
             "example": {
                 "detections": [
                     {
                         "label": "evse",
                         "confidence": 0.87,
                         "bbox": [150, 200, 35, 30]
                     }
                 ],
                 "image_url": "http://127.0.0.1:8000/static/results/unique_image_name.png",
                 "processing_time": 2.45,
                 "mode": "fast"
             }
         }

# --- Dependency to get the detector instance ---
async def get_detector() -> BlueprintDetector:
    detector = ml_models.get("blueprint_detector")
    if detector is None:
        raise HTTPException(
             status_code=503,
             detail="Object detection model is not available."
        )
    return detector

# --- Helper Function for BBox Transformation ---
def transform_detection_results(raw_detections: List) -> List[Dict]:
    transformed_detections = []
    if not raw_detections:
        return []
    print(f"Transforming {len(raw_detections)} detections...")

    # Check if the detections are in the new format (list of lists) or old format (list of dicts)
    if raw_detections and isinstance(raw_detections[0], list):
        # New format from detect.py: [x1, y1, x2, y2, conf, class_id]
        for det in raw_detections:
            try:
                x1, y1, x2, y2, conf, class_id = det
                class_name = ml_models["blueprint_detector"].class_names[int(class_id)]
                transformed_detections.append({
                    "label": class_name,
                    "confidence": float(conf),
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                })
            except Exception as e:
                print(f"Warning: Skipping detection due to error: {e}. Data: {det}")
    else:
        # Old format from original endpoint
        for i, det in enumerate(raw_detections):
            try:
                x1, y1, x2, y2 = det['bbox']
                transformed_bbox = [x1, y1, x2 - x1, y2 - y1]
                transformed_detections.append({
                    "label": det['class_name'],
                    "confidence": det['confidence'],
                    "bbox": transformed_bbox
                })
            except KeyError as e:
                print(f"Warning: Skipping detection {i} due to missing key: {e}. Data: {det}")
            except Exception as e:
                print(f"Warning: Skipping detection {i} due to error: {e}. Data: {det}")

    return transformed_detections

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    return HTMLResponse("<html><body><h1>Blueprint Detector API</h1><p>Send POST to /detect</p></body></html>")

@app.post(
    "/detect",
    response_model=ApiResponse,
    summary="Detect Symbols & Get Image URL",
    description="Upload a blueprint image (PNG or JPG). Returns JSON with detections and a URL to the annotated image. You can choose between 'fast' mode (default) for quick inference or 'accurate' mode for higher accuracy.",
    tags=["Detection"]
)
async def detect_symbols_and_get_url(
    request: Request,
    file: UploadFile = File(..., description="Blueprint image file (PNG or JPG)."),
    mode: str = Query("fast", description="Detection mode: 'fast' (default) or 'accurate'"),
    detector: BlueprintDetector = Depends(get_detector)
):
    if file.content_type not in ["image/png", "image/jpeg"]:
        raise HTTPException(
             status_code=400,
             detail=f"Invalid file type: {file.content_type}. Please upload PNG/JPG."
        )

    if mode not in ["fast", "accurate"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode: {mode}. Choose either 'fast' or 'accurate'."
        )

    start_time = time.time()
    tmp_file_path = None
    saved_image_path = None
    image_url = None
    raw_detections = []
    annotated_image = None # Initialize variable

    try:
        image_content = await file.read()
        await file.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_file_path = tmp_file.name
            print(f"Uploaded file saved temporarily to: {tmp_file_path}")

        print(f"Running detection on {tmp_file_path} in {mode} mode...")

        # Choose detection parameters based on mode
        if mode == "fast":
            # Fast mode: single scale detection
            raw_detections = detector.detect_tiled(
                image_path=tmp_file_path,
                tile_size=640,
                overlap=0.3,
                nms_threshold=0.45
            )
        else:
            # Accurate mode: multi-scale detection with flips
            # Using settings that reduce duplicate boxes while maintaining accuracy
            tiles = [640, 960, 1280] if cv2.imread(tmp_file_path).shape[0] > 1000 else [640, 960]
            overlaps = [0.5] * len(tiles)

            all_detections = []

            # Run detection at each scale
            for i, (tile_size, overlap) in enumerate(zip(tiles, overlaps)):
                print(f"Processing scale {i+1}/{len(tiles)}: tile size {tile_size}px")

                # Original orientation
                detections = detector.detect_tiled(
                    image_path=tmp_file_path,
                    tile_size=tile_size,
                    overlap=overlap,
                    nms_threshold=0.6,
                )
                all_detections.extend(detections)

                # Horizontal flip for test-time augmentation
                if True:  # Always use flips in accurate mode
                    img = cv2.imread(tmp_file_path)
                    img_h_flip = cv2.flip(img, 1)  # 1 for horizontal flip
                    flip_path = f"{tmp_file_path}_flip.jpg"
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
                        det[0] = img_width - det[2]  # x1 becomes width - x2
                        det[2] = img_width - det[0]  # x2 becomes width - x1

                    all_detections.extend(flip_detections)
                    os.remove(flip_path)

            # First standardize boxes to fix aspect ratio issues
            standardized_detections = []
            max_aspect_ratio = 3.0
            for box in all_detections:
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
                    standardized_detections.append([new_x1, y1, new_x2, y2, conf, class_id])
                else:
                    standardized_detections.append(box)

            # Apply final NMS to remove duplicates
            raw_detections = apply_nms(standardized_detections, 0.4)
            print(f"Multi-scale detection complete. Found {len(raw_detections)} symbols after NMS.")

        # --- Log raw detections ---
        print(f"== RAW DETECTIONS ({len(raw_detections)}) ==")
        for i, det in enumerate(raw_detections):
             print(f"  {i}: {det}")
        print("== END RAW DETECTIONS ==")

        print("Generating annotated image...")
        try:
            img_np = cv2.imdecode(np.frombuffer(image_content, np.uint8), cv2.IMREAD_COLOR)
            if img_np is not None:
                # Use the draw_detections method from the detector
                annotated_image = detector.draw_detections(img_np, raw_detections)
                if annotated_image is not None:
                    unique_id = str(uuid.uuid4())[:8]
                    filename = f"{unique_id}_{Path(file.filename).stem}_detected.png"
                    saved_image_path = RESULTS_DIR_PATH / filename
                    cv2.imwrite(str(saved_image_path), annotated_image)
                    print(f"Saved annotated image to {saved_image_path}")

                    # Create URL that will be accessible
                    image_url = f"{request.url.scheme}://{request.url.netloc}/{STATIC_DIR_NAME}/{RESULTS_SUBDIR}/{filename}"
                else:
                    print("Warning: draw_detections returned None")
            else:
                print("Warning: Failed to decode image")
        except Exception as e:
            print(f"Error generating annotated image: {e}")
            # Continue without image

        # Process detections for API response
        transformed_detections = transform_detection_results(raw_detections)

        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds.")

        return ApiResponse(
            detections=transformed_detections,
            image_url=image_url,
            processing_time=round(processing_time, 2),
            mode=mode
        )

    except Exception as e:
        print(f"Error in detect_symbols: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
                print(f"Removed temporary file: {tmp_file_path}")
            except Exception as e:
                print(f"Warning: Failed to remove temporary file: {e}")

# --- Run with Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server with Uvicorn...")
    print("Access the API at http://127.0.0.1:8000")
    print("API Docs available at http://127.0.0.1:8000/docs")
    print(f"Serving annotated images from: {RESULTS_DIR_PATH}")
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
