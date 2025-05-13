#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

# --- Add Project Root to sys.path --- START ---
# Get the directory of the current script (train/)
script_dir = Path(__file__).resolve().parent
# Get the parent directory (project root)
project_root = script_dir.parent
# Add the project root to the system path
sys.path.insert(0, str(project_root))
print(f"Added project root to sys.path: {project_root}") # Debug print
# --- Add Project Root to sys.path --- END ---

# Now the import should work
from app.utils.pdf_utils import convert_pdf_to_image, save_image

# --- Configuration ---
# Use paths relative to the project root for better portability
DEFAULT_PDF_INPUT_DIR = "data/raw"
DEFAULT_OUTPUT_DIR = "data/images"
DEFAULT_DPI = 300
# Pages to extract (0-indexed, e.g., [0, 1] for first and second pages)
PAGES_TO_EXTRACT = [0, 1] # Corresponds to E003 (page 1) and E004 (page 2) if PDF is 0-indexed
# --- End Configuration ---

def main(args):
    # --- Input PDF Handling ---
    pdf_input_dir = Path(args.pdf_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    dpi = args.dpi
    pages_to_extract = [int(p) for p in args.pages.split(',')] if args.pages else PAGES_TO_EXTRACT

    # Find the PDF file (assuming only one relevant PDF in raw)
    pdf_files = list(pdf_input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {pdf_input_dir}")
        print("Please place your blueprint PDF files in that directory.")
        sys.exit(1)
    
    # For simplicity, using the first PDF found. Modify if multiple PDFs need specific handling.
    pdf_path = pdf_files[0]
    print(f"Using PDF: {pdf_path}")

    # --- Output Directory Handling ---
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory set to: {output_dir}")
    except OSError as e:
        print(f"Error: Could not create output directory {output_dir}. Reason: {e}")
        sys.exit(1)

    # --- PDF Processing ---
    print(f"Processing PDF: {pdf_path} with DPI: {dpi}...")
    print(f"Extracting pages: {[p + 1 for p in pages_to_extract]} (0-indexed: {pages_to_extract})")
    
    extracted_pages_paths = []
    
    for page_num in pages_to_extract:
        try:
            print(f"Processing page {page_num + 1}...")
            image_array = convert_pdf_to_image(str(pdf_path), page_number=page_num, dpi=dpi)
            
            output_image_name = f"{pdf_path.stem}_page{page_num + 1}_dpi{dpi}.png"
            output_image_path = output_dir / output_image_name
            
            save_image(image_array, str(output_image_path))
            
            extracted_pages_paths.append(str(output_image_path))
            print(f"  -> Saved to: {output_image_path}")
            
        except ValueError as ve:
            print(f"Error processing page {page_num + 1}: {ve}")
            continue
        except Exception as e:
            print(f"Unexpected error processing page {page_num + 1}: {e}")
            continue
    
    if extracted_pages_paths:
        print(f"\nSuccessfully extracted {len(extracted_pages_paths)} page(s):")
        for path in extracted_pages_paths:
            print(f"  - {path}")
    else:
        print("No pages were successfully extracted.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract pages from a blueprint PDF.")
    parser.add_argument("--pdf_dir", type=str, default=os.path.join(project_root, DEFAULT_PDF_INPUT_DIR), 
                        help="Directory containing the input PDF file(s).")
    parser.add_argument("--output_dir", type=str, default=os.path.join(project_root, DEFAULT_OUTPUT_DIR), 
                        help="Directory to save extracted PNG images.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, 
                        help="DPI for rendering PDF pages.")
    parser.add_argument("--pages", type=str, default=",".join(map(str, PAGES_TO_EXTRACT)),
                        help="Comma-separated list of 0-indexed page numbers to extract (e.g., '0,1').")
    
    args = parser.parse_args()
    main(args)
