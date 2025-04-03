import os
import logging
import pytesseract
from PIL import Image
import pdf2image
import io
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    # Convert the image to grayscale
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get rid of background noise
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply dilation to make text more solid
    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return Image.fromarray(dilated)

def extract_text_from_image(image_path):
    """Extract text from an image file using PyTesseract OCR"""
    try:
        image = Image.open(image_path)
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Extract text using PyTesseract
        text = pytesseract.image_to_string(processed_image)
        logger.debug(f"Extracted text from image: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PDF2Image and PyTesseract"""
    try:
        # Convert PDF to list of images
        pages = pdf2image.convert_from_path(pdf_path)
        logger.debug(f"Converted PDF with {len(pages)} pages")
        
        full_text = ""
        for i, page in enumerate(pages):
            # Preprocess the image
            processed_page = preprocess_image(page)
            
            # Extract text from the page
            text = pytesseract.image_to_string(processed_page)
            full_text += f"\n--- Page {i+1} ---\n{text}\n"
        
        logger.debug(f"Extracted text from PDF: {len(full_text)} characters")
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def process_file(file_path):
    """Process an uploaded file (PDF or image) to extract text"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        return extract_text_from_image(file_path)
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        return ""

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
    """
    # Simple segmentation strategy: split by lines and look for question indicators
    lines = text.split('\n')
    segments = []
    current_segment = ""
    current_question = 1
    
    for line in lines:
        # Check for question number indicators
        # This is a simple approach and may need customization for different formats
        question_indicators = [
            f"Q{current_question}", 
            f"Question {current_question}", 
            f"{current_question}.", 
            f"{current_question})"
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
    
    # If we couldn't find the right number of segments, use a simpler approach
    if len(segments) != num_questions:
        logger.warning(f"Could not identify {num_questions} segments automatically. Using simple text division.")
        
        # Simple division of text into roughly equal parts
        text_length = len(text)
        chunk_size = text_length // num_questions
        segments = []
        
        for i in range(num_questions):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, text_length)
            segments.append(text[start:end])
    
    return segments[:num_questions]  # Ensure we only return up to num_questions segments
