import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
import tempfile

def convert_pdf_to_image(pdf_path: str, page_number: int = 0, dpi: int = 150) -> np.ndarray:
    """
    Convert a specific page of a PDF to an image (NumPy array) at the specified DPI.
    
    Args:
        pdf_path: Absolute path to the PDF file.
        page_number: The page number to convert (0-indexed).
        dpi: Resolution in dots per inch for the output image (min 150).
        
    Returns:
        NumPy array representing the image in RGB format.

    Raises:
        FileNotFoundError: If the pdf_path does not exist.
        ValueError: If the PDF has no pages, the page_number is out of range, or DPI is invalid.
        Exception: For other MuPDF or image conversion errors.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    if not isinstance(dpi, int) or dpi < 150:
        # Enforce a minimum DPI for quality, can be adjusted.
        # print(f"Warning: DPI {dpi} is too low or invalid. Setting to 150.")
        dpi = 150 
    
    zoom = dpi / 72.0  # PDF default DPI is 72

    try:
    doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            raise ValueError(f"The PDF '{os.path.basename(pdf_path)}' has no pages.")
        
        if not (0 <= page_number < len(doc)):
            doc.close()
            raise ValueError(f"Page number {page_number} is out of range for PDF with {len(doc)} pages.")
            
        page = doc.load_page(page_number)
        
        # Create a matrix for rendering at the desired DPI
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render page to a pixmap (pixel map)
        pix = page.get_pixmap(matrix=matrix, alpha=False) # alpha=False for RGB
        
        # Convert pixmap to a NumPy array
        # pix.samples is a bytes object: (width, height, n_components)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        doc.close() # Close the document

        # Ensure the image is in RGB format
        if pix.n == 4:  # RGBA
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:  # Grayscale
            img_array_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif pix.n == 3: # RGB
            img_array_rgb = img_array
        else:
            raise ValueError(f"Unsupported number of color components: {pix.n} on page {page_number}")
            
        return img_array_rgb
    
    except Exception as e:
        # Catch any other MuPDF or processing errors
        if 'doc' in locals() and doc: # Ensure doc is closed if opened
            doc.close()
        # Re-raise with more context if it's a known ValueError, otherwise wrap generic error
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise Exception(f"Failed to convert PDF page {page_number} to image. Reason: {e}")


def save_image(image_array: np.ndarray, output_path: str) -> str:
    """
    Save a NumPy array image to disk.

    Args:
        image_array: Image as a NumPy array (expected in RGB format).
        output_path: Absolute path to save the image (e.g., /path/to/image.png).

    Returns:
        The absolute path where the image was saved.

    Raises:
        OSError: If the directory cannot be created.
        cv2.error: If OpenCV fails to save the image.
    """
    output_path_obj = Path(output_path).resolve() # Ensure absolute path

    # Create the output directory if it doesn't exist
    try:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Could not create directory {output_path_obj.parent}. Reason: {e}")
    
    try:
        # OpenCV expects BGR format for writing, so convert if it's RGB
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Save image (PNG format for lossless quality)
        # Use high compression for PNG. For JPEG, use [cv2.IMWRITE_JPEG_QUALITY, 95]
        cv2.imwrite(str(output_path_obj), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]) # 0 for best compression
        return str(output_path_obj)
    except cv2.error as e:
        raise cv2.error(f"OpenCV failed to save image to {output_path_obj}. Reason: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while saving image to {output_path_obj}. Reason: {e}") 