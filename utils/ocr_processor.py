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
        # Convert PDF to images, processing one page at a time to save memory
        # Set dpi lower to save memory but still get readable text
        logger.debug(f"Processing PDF: {pdf_path}")
        
        # Try to get the number of pages
        info = pdf2image.pdfinfo_from_path(pdf_path)
        num_pages = info["Pages"]
        logger.debug(f"PDF contains {num_pages} pages")
        
        full_text = ""
        # Process one page at a time instead of loading all pages at once
        for page_num in range(1, num_pages + 1):
            try:
                # Lower dpi to save memory
                page = pdf2image.convert_from_path(
                    pdf_path, 
                    first_page=page_num, 
                    last_page=page_num,
                    dpi=150,  # Lower DPI
                    thread_count=1,  # Single thread
                    use_cropbox=True,
                    grayscale=True  # Use grayscale to reduce memory
                )
                
                if page and len(page) > 0:
                    # Preprocess the image
                    processed_page = preprocess_image(page[0])
                    
                    # Extract text from the page
                    text = pytesseract.image_to_string(processed_page)
                    full_text += f"\n--- Page {page_num} ---\n{text}\n"
                    
                    # Free memory
                    del page
                    del processed_page
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
    try:
        # Remove any page markers from OCR
        text = re.sub(r'--- Page \d+ ---', '', text)
        
        # Try multiple segmentation strategies
        logger.debug(f"Attempting to segment text into {num_questions} answers")
        
        # Strategy 1: Split by question number patterns (Q1, Q2, Question 1, etc.)
        segments = segment_by_question_markers(text, num_questions)
        
        # Strategy 2: If first strategy failed, try to identify segments by looking for 
        # any number followed by a period or parenthesis
        if len(segments) != num_questions:
            logger.debug("First segmentation strategy failed, trying alternate pattern matching")
            segments = segment_by_number_indicators(text, num_questions)
        
        # Strategy 3: If still unsuccessful, try to split by paragraph breaks
        if len(segments) != num_questions:
            logger.debug("Second segmentation strategy failed, trying paragraph separation")
            segments = segment_by_paragraphs(text, num_questions)
        
        # Final fallback: If all else fails, just divide the text equally
        if len(segments) != num_questions:
            logger.warning(f"All segmentation strategies failed. Using equal division of text.")
            segments = segment_equally(text, num_questions)
            
        # Ensure we return exactly num_questions segments
        if len(segments) < num_questions:
            # If we have too few segments, duplicate the last one
            last_segment = segments[-1] if segments else "No text available"
            while len(segments) < num_questions:
                segments.append(f"{last_segment} (continued)")
        
        return segments[:num_questions]
        
    except Exception as e:
        logger.error(f"Error in text segmentation: {e}")
        # Return basic segments if all else fails
        return ["Error segmenting text"] * num_questions

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
