import os
import re
import logging
import pytesseract
from PIL import Image
import pdf2image
import io
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import fitz  # PyMuPDF
import base64
import requests
from google.cloud import vision
import json

# Setup logging
logger = logging.getLogger(__name__)

def preprocess_image(image, is_handwritten=False):
    """
    Google Lens-style preprocess image for better OCR results

    Args:
        image: PIL Image object
        is_handwritten: Boolean flag to indicate if the image contains handwritten text

    Returns:
        Tuple of:
        - Processed PIL Image for OCR
        - PIL Image with bounding boxes visualized (for display)
        - List of dictionaries with text region information
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        original_img = img_array.copy()  # Keep a copy for visualization

        # Check if the image is already grayscale
        if len(img_array.shape) == 2:
            gray = img_array  # Already grayscale
            # Convert to 3-channel for visualization
            original_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            gray = img_array[:, :, 0]  # Single channel
            original_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            # Convert to grayscale for processing
            if img_array.shape[2] == 3:  # RGB
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                original_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif img_array.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
                original_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                gray = img_array[:, :, 0]

        # Create a structuring element for morphological operations
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))

        if is_handwritten:
            # Enhanced processing for handwritten text

            # Apply bilateral filter - preserves edges better while removing noise
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)

            # Apply Gaussian blur to further reduce noise
            blurred = cv2.GaussianBlur(filtered, (5, 5), 0)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)

            # Apply adaptive thresholding - better for handwritten text with varying intensity
            thresh = cv2.adaptiveThreshold(
                enhanced, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11,  # Block size
                2    # Constant subtracted from mean
            )

            # Apply morphological operations to clean up the image
            # Create a kernel for morphological operations
            kernel = np.ones((2, 2), np.uint8)

            # Perform closing operation to fill small gaps in text
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Perform opening to remove small noise
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

            # Google Lens style: Detect text regions using morphological operations
            # Dilate to connect characters in text lines
            dilated = cv2.dilate(opened, rect_kernel, iterations=1)

            # Invert back to black text on white background for OCR
            processed = cv2.bitwise_not(opened)
        else:
            # Standard processing for printed text
            # Apply thresholding to get rid of background noise
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply dilation to connect characters in text lines
            dilated = cv2.dilate(thresh, rect_kernel, iterations=1)

            # Apply dilation to make text more solid for OCR
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.dilate(thresh, kernel, iterations=1)

        # Find contours - this is where we detect text regions like Google Lens
        # We use the dilated image which has connected text components
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the original for visualization
        visualization_img = original_img.copy()

        # Process text regions
        text_regions = []
        region_id = 1

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out very small regions that are likely noise
            if w < 15 or h < 10 or w * h < 100:
                continue

            # Filter out regions with extreme aspect ratios
            aspect_ratio = w / float(h) if h > 0 else 0
            if aspect_ratio > 15 or aspect_ratio < 0.1:
                continue

            # Draw rectangle on visualization image
            cv2.rectangle(visualization_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add region ID text
            cv2.putText(visualization_img, f"#{region_id}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Store region information
            text_regions.append({
                'id': region_id,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio,
            })

            region_id += 1

        # Sort regions top-to-bottom for reading order
        text_regions.sort(key=lambda r: r['y'])

        # Return processed image for OCR, visualization image, and region information
        return Image.fromarray(processed), Image.fromarray(cv2.cvtColor(visualization_img, cv2.COLOR_BGR2RGB)), text_regions

    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        # Return original images if preprocessing fails
        if isinstance(image, Image.Image):
            return image, image, []
        else:
            # Convert numpy array to PIL if needed
            try:
                pil_image = Image.fromarray(np.uint8(image))
                return pil_image, pil_image, []
            except:
                # Create a blank image as fallback
                blank = Image.new('RGB', (100, 100), color='white')
                return blank, blank, []

def extract_text_with_google_vision(image_path):
    """
    Extract text using Google Vision API for better handwritten text recognition
    """
    try:
        # Check if Google Vision client can be initialized
        try:
            client = vision.ImageAnnotatorClient()
            logger.info("Using Google Vision API for OCR")
        except Exception as vision_error:
            logger.warning(f"Google Vision API not available: {vision_error}")
            return None

        # Read the image file
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        # Create Vision API image object
        image = vision.Image(content=content)

        # Detect text in the image using document text detection (better for handwriting)
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            logger.error(f'Google Vision API error: {response.error.message}')
            return None

        # Extract full text
        full_text = response.full_text_annotation.text if response.full_text_annotation else ""
        
        # Also get detailed text annotations for better structure
        texts = response.text_annotations
        detailed_text = ""
        
        if texts:
            # First annotation contains the full detected text
            detailed_text = texts[0].description
            
        # Use the more complete result
        result_text = detailed_text if len(detailed_text) > len(full_text) else full_text
        
        if result_text:
            logger.info(f"Google Vision API extracted {len(result_text)} characters")
            return f"[Google Vision API - Handwriting Optimized]\n{result_text}"
        else:
            logger.warning("Google Vision API returned empty result")
            return None
            
    except Exception as e:
        logger.error(f"Google Vision API error: {e}")
        return None

def extract_text_with_azure_vision(image_path):
    """
    Extract text using Azure Computer Vision API for handwritten text
    """
    try:
        # Azure Computer Vision endpoint and key (should be set as environment variables)
        endpoint = os.getenv('AZURE_VISION_ENDPOINT')
        key = os.getenv('AZURE_VISION_KEY')
        
        if not endpoint or not key:
            logger.warning("Azure Vision API credentials not found")
            return None
            
        logger.info("Using Azure Computer Vision API for OCR")
        
        # Prepare the image
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Azure OCR endpoint
        ocr_url = f"{endpoint}/vision/v3.2/read/analyze"
        
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Content-Type': 'application/octet-stream'
        }
        
        # Submit image for analysis
        response = requests.post(ocr_url, headers=headers, data=image_data)
        
        if response.status_code != 202:
            logger.error(f"Azure Vision API request failed: {response.status_code}")
            return None
        
        # Get operation location from response headers
        operation_location = response.headers.get('Operation-Location')
        if not operation_location:
            logger.error("No operation location in Azure response")
            return None
        
        # Poll for results
        import time
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            time.sleep(2)  # Wait 2 seconds between polls
            
            result_response = requests.get(operation_location, headers={'Ocp-Apim-Subscription-Key': key})
            
            if result_response.status_code == 200:
                result = result_response.json()
                
                if result['status'] == 'succeeded':
                    # Extract text from results
                    extracted_text = ""
                    
                    if 'analyzeResult' in result and 'readResults' in result['analyzeResult']:
                        for page in result['analyzeResult']['readResults']:
                            for line in page.get('lines', []):
                                extracted_text += line['text'] + '\n'
                    
                    if extracted_text:
                        logger.info(f"Azure Vision API extracted {len(extracted_text)} characters")
                        return f"[Azure Computer Vision - Handwriting Optimized]\n{extracted_text}"
                    else:
                        logger.warning("Azure Vision API returned empty result")
                        return None
                        
                elif result['status'] == 'failed':
                    logger.error("Azure Vision API analysis failed")
                    return None
                    
            attempt += 1
        
        logger.error("Azure Vision API polling timeout")
        return None
        
    except Exception as e:
        logger.error(f"Azure Vision API error: {e}")
        return None

def extract_text_from_image(image_path):
    """
    Extract text from an image file using PyTesseract OCR with Google Lens-like 
    region detection for better results
    """
    try:
        logger.info(f"Extracting text from image: {image_path}")

        # Create uploads directory for saving visualization if it doesn't exist
        viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Load and resize the image to save memory
        try:
            image = Image.open(image_path)

            # Resize large images to reduce memory usage
            width, height = image.size
            if width > 1200 or height > 1200:
                # Calculate new dimensions while maintaining aspect ratio
                if width > height:
                    new_width = 1200
                    new_height = int(height * (new_width / width))
                else:
                    new_height = 1200
                    new_width = int(width * (new_height / height))

                logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height} to save memory")
                image = image.resize((new_width, new_height))
        except Exception as img_err:
            logger.error(f"Error loading or resizing image: {img_err}")
            return f"[Error loading image: {str(img_err)}]"

        # First try cloud-based OCR APIs for better handwritten text recognition
        cloud_result = None
        
        # Try multiple cloud OCR services in order of preference
        # 1. Google Vision API (most accurate for handwriting)
        cloud_result = extract_text_with_google_vision(image_path)
        
        # 2. If Google Vision fails, try Azure
        if not cloud_result:
            cloud_result = extract_text_with_azure_vision(image_path)
        
        # 3. If both fail, try OCR.Space (free option)
        if not cloud_result:
            try:
                from config_ocr import extract_text_with_ocr_space
                cloud_result = extract_text_with_ocr_space(image_path)
            except ImportError:
                logger.warning("config_ocr module not available for OCR.Space integration")
                cloud_result = None
        
        # If cloud APIs succeeded and returned substantial text, use that result
        if cloud_result and len(cloud_result.strip()) > 50:
            logger.info("Using cloud-based OCR result for better accuracy")
            
            # Still create visualization for user feedback
            try:
                processed_image, visualization_img, text_regions = preprocess_image(image, is_handwritten=True)
                base_filename = os.path.basename(image_path)
                viz_filename = f"viz_{base_filename.split('.')[0]}.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)
                visualization_img.save(viz_path)
                
                visualization_url = f"/static/visualizations/{viz_filename}"
                final_result = f"[Cloud OCR - Enhanced Handwriting Recognition. View regions <a href='{visualization_url}' target='_blank'>here</a>]\n\n"
                final_result += cloud_result
                
                return final_result
            except Exception as viz_err:
                logger.warning(f"Visualization creation failed: {viz_err}")
                return cloud_result
        
        # Fallback to local processing with enhanced settings
        logger.info("Cloud OCR not available or insufficient, using enhanced local OCR")
        
        # Process with Google Lens-like region detection
        try:
            # Process with autodetection - returns processed image, visualization, and text regions
            processed_image, visualization_img, text_regions = preprocess_image(image, is_handwritten=True)

            # Save visualization image for display
            base_filename = os.path.basename(image_path)
            viz_filename = f"viz_{base_filename.split('.')[0]}.jpg"
            viz_path = os.path.join(viz_dir, viz_filename)
            visualization_img.save(viz_path)
            logger.info(f"Saved visualization to {viz_path}")

            # If no text regions found, try again with printed text settings
            if len(text_regions) == 0:
                logger.info("No text regions detected with handwritten settings, trying printed text mode")
                processed_image, visualization_img, text_regions = preprocess_image(image, is_handwritten=False)

                # Save updated visualization
                visualization_img.save(viz_path)

            # Google Lens style: Process each text region separately for better accuracy
            full_text = ""

            if text_regions:
                # Use OCR on each detected region for better accuracy
                # OEM 1: Legacy Tesseract engine (less memory intensive)
                # PSM 6: Assume a single uniform block of text
                # Optimize Tesseract for handwriting recognition
                custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!? -c textord_heavy_nr=1 -c textord_min_linesize=3'

                # Convert processed image to numpy for cropping regions
                processed_np = np.array(processed_image)

                for region in text_regions:
                    try:
                        # Extract region coordinates
                        x, y, w, h = region['x'], region['y'], region['width'], region['height']

                        # Ensure coordinates are within image bounds
                        if y >= processed_np.shape[0] or x >= processed_np.shape[1]:
                            continue

                        # Add padding around the region for better OCR results
                        pad = 5
                        y1 = max(0, y - pad)
                        y2 = min(processed_np.shape[0], y + h + pad)
                        x1 = max(0, x - pad)
                        x2 = min(processed_np.shape[1], x + w + pad)

                        # Extract region
                        region_img = processed_np[y1:y2, x1:x2]

                        # Skip empty regions
                        if region_img.size == 0:
                            continue

                        # Convert region to PIL image
                        region_pil = Image.fromarray(region_img)

                        # Extract text from this region
                        region_text = pytesseract.image_to_string(
                            region_pil,
                            config=custom_config,
                            lang='eng',
                            timeout=10
                        )

                        # Add region text to full text with region identifier
                        if region_text.strip():
                            full_text += f"Region #{region['id']} ({x},{y}): {region_text.strip()}\n\n"
                    except Exception as region_err:
                        logger.error(f"Error processing region {region['id']}: {region_err}")

                # If region-based extraction yielded too little text, try whole image
                if len(full_text.strip()) < 20:
                    logger.info("Region-based text detection yielded little text. Trying whole image.")
                    whole_image_text = pytesseract.image_to_string(
                        processed_image,
                        config=custom_config,
                        lang='eng',
                        timeout=30
                    )

                    if len(whole_image_text.strip()) > len(full_text.strip()):
                        full_text = whole_image_text
            else:
                # If no regions detected, process the entire image
                logger.info("No text regions detected, processing entire image")
                # Optimize Tesseract for handwriting recognition
                custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!? -c textord_heavy_nr=1 -c textord_min_linesize=3'
                full_text = pytesseract.image_to_string(
                    processed_image,
                    config=custom_config,
                    lang='eng',
                    timeout=30
                )

            # If still not getting good results, try with printed text mode
            if len(full_text.strip()) < 20:
                logger.info("Handwritten mode yielded little text. Trying printed text processing.")
                processed_image, viz2, regions2 = preprocess_image(image, is_handwritten=False)

                # Save new visualization
                viz2.save(viz_path)

                alt_text = pytesseract.image_to_string(
                    processed_image,
                    config=r'--oem 1 --psm 6',
                    lang='eng',
                    timeout=30
                )

                # Use the better result
                if len(alt_text.strip()) > len(full_text.strip()):
                    logger.info("Printed text processing yielded better results.")
                    full_text = alt_text

            # Clean up resources
            del processed_image
            del image

            # Add visualization info to the output
            visualization_url = f"/static/visualizations/{viz_filename}"
            result_text = f"[GoogleLens-style text detection enabled. View detected regions <a href='{visualization_url}' target='_blank'>here</a>]\n\n"
            result_text += full_text

            logger.debug(f"Extracted text from image: {len(full_text)} characters from {len(text_regions)} regions")
            return result_text

        except Exception as ocr_err:
            logger.error(f"OCR error: {ocr_err}")
            return f"[OCR processing error: {str(ocr_err)}]"

    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return f"[Error processing image: {str(e)}]"

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PDF2Image and PyTesseract"""
    try:
        # Convert PDF to images, processing one page at a time to save memory
        # Set dpi lower to save memory but still get readable text
        logger.debug(f"Processing PDF: {pdf_path}")

        # First try direct text extraction (much faster and less memory intensive if possible)
        try:
            # Using pdfminer.six for direct text extraction
            from pdfminer.high_level import extract_text
            from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
            from pdfminer.converter import TextConverter
            from pdfminer.layout import LAParams
            from pdfminer.pdfpage import PDFPage
            from io import StringIO

            # Extract text directly from PDF using pdfminer
            extracted_text = extract_text(pdf_path)

            if extracted_text and len(extracted_text.strip()) > 100:
                logger.info(f"Successfully extracted text directly from PDF, skipping OCR")
                return extracted_text

            # If direct extraction failed, try with more detailed settings
            with open(pdf_path, 'rb') as pdf_file:
                resource_manager = PDFResourceManager()
                output_string = StringIO()
                codec = 'utf-8'
                laparams = LAParams()
                converter = TextConverter(resource_manager, output_string, codec=codec, laparams=laparams)
                interpreter = PDFPageInterpreter(resource_manager, converter)

                for page in PDFPage.get_pages(pdf_file, check_extractable=False):
                    interpreter.process_page(page)

                alternative_text = output_string.getvalue()

                if alternative_text and len(alternative_text.strip()) > 100:
                    logger.info(f"Successfully extracted text with alternative method, skipping OCR")
                    return alternative_text

            logger.info("Direct text extraction yielded insufficient results, proceeding with OCR")
        except Exception as pdf_ex:
            logger.info(f"Direct text extraction failed: {pdf_ex}. Proceeding with OCR.")

        # Try to get the number of pages
        info = pdf2image.pdfinfo_from_path(pdf_path)
        num_pages = info["Pages"]
        logger.debug(f"PDF contains {num_pages} pages")

        # Limit page processing to avoid memory issues
        max_pages = min(num_pages, 10)  # Process at most 10 pages to avoid memory issues
        if num_pages > max_pages:
            logger.warning(f"PDF has {num_pages} pages, limiting processing to first {max_pages} pages")

        full_text = ""
        # Process one page at a time instead of loading all pages at once
        for page_num in range(1, max_pages + 1):
            try:
                # Lower dpi to save memory
                page = pdf2image.convert_from_path(
                    pdf_path, 
                    first_page=page_num, 
                    last_page=page_num,
                    dpi=100,  # Even lower DPI to save memory
                    thread_count=1,  # Single thread
                    use_cropbox=True,
                    grayscale=True,  # Use grayscale to reduce memory
                    size=(800, None)  # Resize to lower resolution
                )

                if page and len(page) > 0:
                    try:
                        # Process with reduced complexity for memory efficiency
                        # Resize the image to reduce memory usage
                        img = page[0]
                        width, height = img.size
                        if width > 1000:
                            # Downscale large images to save memory
                            new_width = 1000
                            new_height = int(height * (new_width / width))
                            img = img.resize((new_width, new_height))

                        # Create visualizations directory if it doesn't exist
                        viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'visualizations')
                        os.makedirs(viz_dir, exist_ok=True)

                        # Preprocess the image with Google Lens-style region detection
                        processed_page, viz_image, text_regions = preprocess_image(img, is_handwritten=True)

                        # Save the visualization image
                        viz_filename = f"viz_pdf_p{page_num}_{os.path.basename(pdf_path).split('.')[0]}.jpg"
                        viz_path = os.path.join(viz_dir, viz_filename)
                        viz_image.save(viz_path)

                        # Process each detected text region separately for better results
                        page_text = ""

                        if text_regions:
                            # Process individual regions
                            # Optimize Tesseract for handwriting recognition
                            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!? -c textord_heavy_nr=1 -c textord_min_linesize=3'  # Memory-efficient config

                            # Convert processed image to numpy for cropping
                            processed_np = np.array(processed_page)

                            # Process each region
                            for region in text_regions:
                                try:
                                    # Extract region coordinates
                                    x, y, w, h = region['x'], region['y'], region['width'], region['height']

                                    # Ensure coordinates are within image bounds
                                    if y >= processed_np.shape[0] or x >= processed_np.shape[1]:
                                        continue

                                    # Add padding around the region for better OCR results
                                    pad = 5
                                    y1 = max(0, y - pad)
                                    y2 = min(processed_np.shape[0], y + h + pad)
                                    x1 = max(0, x - pad)
                                    x2 = min(processed_np.shape[1], x + w + pad)

                                    # Extract region
                                    region_img = processed_np[y1:y2, x1:x2]

                                    # Skip empty regions
                                    if region_img.size == 0:
                                        continue

                                    # Convert region to PIL image
                                    region_pil = Image.fromarray(region_img)

                                    # Extract text from this region
                                    region_text = pytesseract.image_to_string(
                                        region_pil,
                                        config=custom_config,
                                        lang='eng',
                                        timeout=10
                                    )

                                    # Add region text to full text with region identifier
                                    if region_text.strip():
                                        page_text += f"Region #{region['id']} ({x},{y}): {region_text.strip()}\n\n"
                                except Exception as region_err:
                                    logger.error(f"Error processing region {region['id']} on page {page_num}: {region_err}")

                            # If region-based extraction yielded little text, try whole page
                            if len(page_text.strip()) < 20:
                                whole_page_text = pytesseract.image_to_string(
                                    processed_page,
                                    config=custom_config,
                                    lang='eng',
                                    timeout=30
                                )

                                if len(whole_page_text.strip()) > len(page_text.strip()):
                                    page_text = whole_page_text
                        else:
                            # If no regions detected, process the entire page
                            # Optimize Tesseract for handwriting recognition
                            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!? -c textord_heavy_nr=1 -c textord_min_linesize=3'
                            page_text = pytesseract.image_to_string(
                                processed_page,
                                config=custom_config,
                                lang='eng',
                                timeout=30
                            )

                        # Add visualization info to the output
                        visualization_url = f"/static/visualizations/{viz_filename}"
                        text = f"[GoogleLens-style text detection enabled. View detected regions for page {page_num} <a href='{visualization_url}' target='_blank'>here</a>]\n\n"
                        text += page_text

                        full_text += f"\n--- Page {page_num} ---\n{text}\n"
                    except Exception as ocr_error:
                        logger.error(f"OCR error on page {page_num}: {ocr_error}")
                        full_text += f"\n--- Page {page_num} ---\n[OCR processing error]\n"
                    finally:
                        # Free memory immediately
                        if 'processed_page' in locals():
                            del processed_page
                        if 'img' in locals():
                            del img
                        if 'page' in locals():
                            del page
                else:
                    logger.warning(f"No image returned for page {page_num}")
            except Exception as page_error:
                logger.error(f"Error processing page {page_num}: {page_error}")
                full_text += f"\n--- Page {page_num} ---\n[Error: Could not process this page]\n"

        if not full_text.strip():
            logger.warning("No text extracted from PDF, trying fallback method")
            # Fallback: try to extract text directly using a simple approach
            with open(pdf_path, 'rb') as pdf_file:
                # This is just checking if there's raw text that can be extracted
                raw_content = pdf_file.read().decode('utf-8', errors='ignore')
                text_content = ' '.join(raw_content.split())
                if len(text_content) > 100:  # If we found some meaningful text
                    full_text = f"[Extracted raw text from PDF]\n{text_content}"

        logger.debug(f"Extracted text from PDF: {len(full_text)} characters")
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        # Return a more explicit error message for troubleshooting
        return f"[Error processing PDF: {str(e)}]"

def process_file(file_path, is_handwritten=True):
    """
    Process an uploaded file (PDF or image) to extract text

    Args:
        file_path: Path to the file to process
        is_handwritten: Boolean indicating if the document contains handwritten text
                      Default is True to enable handwritten text optimizations

    Returns:
        Extracted text from the file
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    # Log the processing attempt
    logger.info(f"Processing file: {file_path} (Handwritten: {is_handwritten})")

    # Process PDF files with multiple extraction methods
    if file_ext == '.pdf':
        try:
            # First try PyMuPDF for direct text extraction (most efficient method)
            logger.info("Using PyMuPDF for PDF text extraction")
            full_text = ""
            page_count = 0

            try:
                # Open the PDF with PyMuPDF
                doc = fitz.open(file_path)

                # Get page count
                page_count = len(doc)
                logger.info(f"PDF contains {page_count} pages")

                # Extract text from each page
                for page_num in range(page_count):
                    try:
                        page = doc.load_page(page_num)
                        page_text = page.get_text()

                        # Enhanced text cleaning for better readability
                        # Remove control characters and non-printable characters
                        page_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', page_text)
                        # Remove excessive whitespace
                        page_text = re.sub(r'\s+', ' ', page_text)
                        # Clean up paragraph breaks for better readability
                        page_text = re.sub(r'(\. |\? |\! )([A-Z])', r'.\n\2', page_text)
                        # Restore proper indentation for paragraphs
                        page_text = re.sub(r'\n', '\n    ', page_text)

                        # Add page marker and text to the result
                        page_marker = f"\n--- Page {page_num + 1} ---\n"
                        full_text += page_marker + page_text.strip() + "\n"
                    except Exception as page_err:
                        logger.error(f"Error extracting text from page {page_num + 1}: {page_err}")
                        full_text += f"\n--- Page {page_num + 1} ---\n[Error: Could not extract text from this page]\n"

                # Close the document
                doc.close()

                # Check if we have meaningful text
                if len(full_text.strip()) > 200:
                    logger.info("Successfully extracted text using PyMuPDF")
                    return full_text

                logger.info("PyMuPDF extraction produced limited text, trying other methods")

            except Exception as pymupdf_err:
                logger.error(f"PyMuPDF extraction error: {pymupdf_err}")
                logger.info("Falling back to alternative extraction methods")

            # Try pdfminer as a second option
            logger.info("Using pdfminer for PDF text extraction")
            try:
                from pdfminer.high_level import extract_text
                extracted_text = extract_text(file_path)

                # Enhanced text cleaning for better readability
                # Remove control characters and non-printable characters
                extracted_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', extracted_text)
                # Remove excessive whitespace while preserving paragraph structure
                extracted_text = re.sub(r'[ \t]+', ' ', extracted_text)
                # Clean up paragraph breaks for better readability
                extracted_text = re.sub(r'(\. |\? |\! )([A-Z])', r'.\n\n\2', extracted_text)
                # Add indentation for paragraphs
                extracted_text = re.sub(r'\n([A-Za-z])', r'\n    \1', extracted_text)

                if len(extracted_text.strip()) > 200:
                    logger.info("Successfully extracted text with pdfminer")

                    # Format the text with page breaks for better readability
                    formatted_text = ""
                    pages = extracted_text.split("\f")  # Form feed character often separates PDF pages

                    for i, page in enumerate(pages):
                        if page.strip():
                            formatted_text += f"\n--- Page {i + 1} ---\n{page.strip()}\n"

                    return formatted_text if formatted_text.strip() else extracted_text
            except Exception as pdfminer_err:
                logger.error(f"pdfminer extraction error: {pdfminer_err}")

            # If direct text extraction methods fail, try our fallback with binary filtering
            try:
                logger.info("Using binary filtering for PDF extraction")
                with open(file_path, 'rb') as pdf_file:
                    # Read content and filter out NULL bytes and other problematic characters
                    raw_content = pdf_file.read()
                    # Filter out NULL bytes before decoding
                    filtered_content = bytearray([b for b in raw_content if b != 0])
                    text_content = filtered_content.decode('utf-8', errors='ignore')

                    # Clean the text further with enhanced processing

                    # 1. Remove control characters except newlines and tabs
                    text_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text_content)

                    # 2. Clean up PDF binary artifacts and formatting markers
                    # Remove PDF binary markers
                    text_content = re.sub(r'%PDF-\d+\.\d+', '', text_content)
                    text_content = re.sub(r'stream[\s\S]*?endstream', '', text_content)
                    text_content = re.sub(r'<<.*?>>', '', text_content)
                    text_content = re.sub(r'obj.*?endobj', '', text_content)
                    text_content = re.sub(r'xref.*?trailer', '', text_content)

                    # 3. Remove excessive blank lines (more than 2 consecutive)
                    text_content = re.sub(r'\n{3,}', '\n\n', text_content)

                    # 4. Remove lines that appear to be PDF binary data or metadata
                    clean_lines = []
                    for line in text_content.split('\n'):
                        # Skip lines that look like binary data or PDF internal format
                        if re.search(r'^[0-9]+ [0-9]+ (obj|R)$', line.strip()) or \
                           re.search(r'^/[A-Za-z]+[/<>]', line.strip()) or \
                           line.strip() in ['stream', 'endstream', 'endobj', 'startxref', '%%EOF'] or \
                           re.match(r'^[\x00-\xff]{1,5}$', line.strip()):
                            continue
                        clean_lines.append(line)

                    # Rejoin the clean lines
                    text_content = '\n'.join(clean_lines)

                    # 5. Add section headers where PDF page breaks might have been
                    # Look for patterns like "Page X" or clear content breaks
                    if 'page' in text_content.lower() or 'answer' in text_content.lower():
                        # Add page markers based on potential page indicators
                        sections = re.split(r'(\bpage\s+\d+\b|\bsection\s+\d+\b|\banswer\s+\d+\b)', 
                                           text_content, flags=re.IGNORECASE)
                        formatted_content = ""
                        for i in range(len(sections)):
                            if i % 2 == 1:  # This is a page/section marker
                                formatted_content += f"\n--- {sections[i].strip().title()} ---\n"
                            else:
                                formatted_content += sections[i]
                        text_content = formatted_content

                    # If we have meaningful content, return it
                    if len(text_content.strip()) > 200:
                        logger.info("Successfully extracted text using enhanced binary filtering")

                        # Final cleanup - ensure the text is human-readable
                        # Remove any remaining PDF artifacts that look like binary content
                        text_content = re.sub(r'[\x80-\xff]{4,}', '', text_content)
                        text_content = re.sub(r'[^\w\s\n.,;:!?()[\]{}\-"\'/#$%&*+<=>@^_`|~]', '', text_content)

                        # Ensure there's a page marker for better display
                        if not re.search(r'---\s*Page', text_content):
                            text_content = "--- Page 1 ---\n" + text_content

                        return f"[Extracted content from {os.path.basename(file_path)}]\n{text_content}"
            except Exception as binary_err:
                logger.error(f"Binary filtering error: {binary_err}")

            # For handwritten content, try OCR as a last resort
            if is_handwritten:
                try:
                    logger.info("Attempting OCR processing for handwritten content")
                    ocr_text = extract_text_from_pdf(file_path)

                    if len(ocr_text.strip()) > 200:
                        logger.info("Successfully extracted text using OCR")
                        return ocr_text
                except Exception as ocr_err:
                    logger.error(f"OCR processing error: {ocr_err}")

            # If all methods fail, return informative message with detected page count
            logger.warning("All extraction methods failed. Returning basic PDF info.")
            return f"[TEXT EXTRACTION FROM {os.path.basename(file_path)}]\n" + \
                   f"This PDF contains {page_count if page_count > 0 else 'multiple'} pages.\n" + \
                   "The system detected this document contains primarily scanned or image-based content.\n" + \
                   "For more accurate results, try uploading a PDF with embedded text."

        except Exception as pdf_err:
            logger.error(f"PDF processing error: {pdf_err}")
            return f"[Error processing PDF: {str(pdf_err)}]"

    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']:
        try:
            # For single images, use optimized OCR with size check
            # Get image dimensions
            from PIL import Image
            img = Image.open(file_path)
            width, height = img.size

            # If the image is very large, resize it to save memory
            if width > 2000 or height > 2000:
                logger.warning(f"Image is large ({width}x{height}), resizing for OCR processing")
                # Calculate new dimensions while maintaining aspect ratio
                if width > height:
                    new_width = 2000
                    new_height = int(height * (new_width / width))
                else:
                    new_height = 2000
                    new_width = int(width * (new_height / height))

                # Create a temporary resized image
                try:
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    # Fallback for older Pillow versions
                    img = img.resize((new_width, new_height), Image.ANTIALIAS)
                temp_path = f"{file_path}_resized.jpg"
                img.save(temp_path)

                # Process the resized image
                result = extract_text_from_image(temp_path)

                # Clean up
                try:
                    os.remove(temp_path)
                except:
                    pass

                return result

            # For smaller images, use the optimized OCR extraction directly
            return extract_text_from_image(file_path)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"[Error processing image: {str(e)}]"
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        return f"[Unsupported file type: {file_ext}]"

def save_uploaded_file(uploaded_file, upload_folder):
    """Save an uploaded file and return the path"""
    filename = secure_filename(uploaded_file.filename)
    file_path = os.path.join(upload_folder, filename)
    uploaded_file.save(file_path)
    logger.debug(f"Saved uploaded file to {file_path}")
    return file_path

def extract_text_segments(text, num_questions):
    """
    Attempt to segment extracted text into different answers
    based on question numbers or patterns

    Args:
        text: The extracted text to segment
        num_questions: Number of questions expected

    Returns:
        List of text segments corresponding to answers
    """
    try:
        # First mark the page boundaries so we can preserve them in the final output
        # Replace page markers with a special token we can restore later
        cleaned_text = re.sub(r'(\n---\s*Page\s+\d+\s*---\s*\n)', r'PAGE_MARKER\1PAGE_MARKER', text)

        # Also handle any PDF-specific formatting issues
        # Replace multiple consecutive newlines with just two
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

        # Remove any remaining control characters
        cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned_text)

        logger.debug(f"Cleaned text for segmentation, original length: {len(text)}, cleaned length: {len(cleaned_text)}")

        # Try multiple segmentation strategies
        logger.debug(f"Attempting to segment text into {num_questions} answers")

        # Strategy 1: Split by question number patterns (Q1, Q2, Question 1, etc.)
        segments = segment_by_question_markers(cleaned_text, num_questions)

        # Strategy 2: If first strategy failed, try to identify segments by looking for 
        # any number followed by a period or parenthesis
        if len(segments) != num_questions:
            logger.debug("First segmentation strategy failed, trying alternate pattern matching")
            segments = segment_by_number_indicators(cleaned_text, num_questions)

        # Strategy 3: If still unsuccessful, try to split by paragraph breaks
        if len(segments) != num_questions:
            logger.debug("Second segmentation strategy failed, trying paragraph separation")
            segments = segment_by_paragraphs(cleaned_text, num_questions)

        # Final fallback: If all else fails, just divide the text equally
        if len(segments) != num_questions:
            logger.warning(f"All segmentation strategies failed. Using equal division of text.")
            segments = segment_equally(cleaned_text, num_questions)

        # Ensure we return exactly num_questions segments
        if len(segments) < num_questions:
            # If we have too few segments, duplicate the last one
            last_segment = segments[-1] if segments else "No text available"
            while len(segments) < num_questions:
                segments.append(f"{last_segment} (continued)")
        elif len(segments) > num_questions:
            # If we have too many segments, combine the extra ones with the last segment
            extra_segments = segments[num_questions:]
            segments = segments[:num_questions-1] + [segments[num_questions-1] + "\n\n" + "\n\n".join(extra_segments)]

        # Clean up each segment
        for i in range(len(segments)):
            # Restore page markers for better readability
            segment_text = segments[i]

            # Replace our special marker with the actual page marker
            if 'PAGE_MARKER' in segment_text:
                # Extract and format page markers
                segment_text = re.sub(r'PAGE_MARKER(\n---\s*Page\s+\d+\s*---\s*\n)PAGE_MARKER', 
                                     r'\1', segment_text)

            # Remove excessive whitespace but preserve newlines for page markers
            segment_text = re.sub(r'[ \t]+', ' ', segment_text).strip()

            # Add a header for better readability
            segments[i] = f"Answer {i+1}:\n{segment_text}"

        return segments[:num_questions]

    except Exception as e:
        logger.error(f"Error in text segmentation: {e}")
        # Return basic segments if all else fails
        return [f"Answer {i+1}:\nError segmenting text" for i in range(num_questions)]

def segment_by_question_markers(text, num_questions):
    """Strategy 1: Look for typical question markers"""
    lines = text.split('\n')
    segments = []
    current_segment = ""
    current_question = 1

    for line in lines:
        # Check for question number indicators
        question_indicators = [
            f"Q{current_question}", 
            f"Question {current_question}", 
            f"{current_question}.", 
            f"{current_question})",
            f"#{current_question}"
        ]

        if any(indicator in line for indicator in question_indicators) and current_segment and current_question <= num_questions:
            segments.append(current_segment.strip())
            current_segment = line
            current_question += 1
        else:
            current_segment += f"\n{line}"

    # Add the last segment
    if current_segment.strip():
        segments.append(current_segment.strip())

    return segments

def segment_by_number_indicators(text, num_questions):
    """Strategy 2: Look for any numbers that might indicate questions"""
    # Pattern to match numbers at the beginning of a line, followed by period/parenthesis
    pattern = r'\n\s*(\d+)[\.\)\:]'

    # Find all matches
    matches = list(re.finditer(pattern, text))

    if not matches:
        return []

    segments = []
    for i in range(len(matches)):
        start_pos = matches[i].start()

        # Determine end position
        if i < len(matches) - 1:
            end_pos = matches[i+1].start()
        else:
            end_pos = len(text)

        # Extract the segment
        segment = text[start_pos:end_pos].strip()
        segments.append(segment)

    return segments

def segment_by_paragraphs(text, num_questions):
    """Strategy 3: Split by paragraph breaks (double newlines)"""
    # Split text by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)

    # Filter out empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]

    if len(paragraphs) < num_questions:
        return []

    # If we have more paragraphs than questions, combine some
    if len(paragraphs) > num_questions:
        # Calculate how many paragraphs per question on average
        paras_per_question = len(paragraphs) // num_questions

        segments = []
        for i in range(num_questions - 1):
            start_idx = i * paras_per_question
            end_idx = (i + 1) * paras_per_question
            segment = "\n\n".join(paragraphs[start_idx:end_idx])
            segments.append(segment)

        # Last segment gets all remaining paragraphs
        last_segment = "\n\n".join(paragraphs[(num_questions - 1) * paras_per_question:])
        segments.append(last_segment)

        return segments

    # If we have exactly the right number of paragraphs
    return paragraphs

def segment_equally(text, num_questions):
    """Final fallback: Just divide the text into equal parts"""
    text_length = len(text)
    chunk_size = text_length // num_questions
    segments = []

    for i in range(num_questions):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, text_length)

        # Try to avoid cutting words
        if i > 0 and start > 0:
            # Look for the nearest space before this position
            space_pos = text.rfind(' ', 0, start)
            if space_pos > start - 50:  # Don't move back too far
                start = space_pos + 1

        if i < num_questions - 1 and end < text_length:
            # Look for the nearest space after this position
            space_pos = text.find(' ', end)
            if space_pos != -1 and space_pos < end + 50:  # Don't move forward too far
                end = space_pos

        segments.append(text[start:end])

    return segments