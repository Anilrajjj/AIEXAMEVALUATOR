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

# Setup logging
logger = logging.getLogger(__name__)

def preprocess_image(image, is_handwritten=False):
    """
    Preprocess image for better OCR results
    
    Args:
        image: PIL Image object
        is_handwritten: Boolean flag to indicate if the image contains handwritten text
    
    Returns:
        Processed PIL Image
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Check if the image is already grayscale
        if len(img_array.shape) == 2:
            gray = img_array  # Already grayscale
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            gray = img_array[:, :, 0]  # Single channel
        else:
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        if is_handwritten:
            # Special processing for handwritten text
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply adaptive thresholding - better for handwritten text with varying intensity
            thresh = cv2.adaptiveThreshold(
                blurred, 
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
            
            # Invert back to black text on white background for OCR
            processed = cv2.bitwise_not(opened)
            
            return Image.fromarray(processed)
        else:
            # Standard processing for printed text
            # Apply thresholding to get rid of background noise
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply dilation to make text more solid
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            return Image.fromarray(dilated)
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        # Return the original image if preprocessing fails
        return image

def extract_text_from_image(image_path):
    """Extract text from an image file using PyTesseract OCR, with memory optimizations"""
    try:
        logger.info(f"Extracting text from image: {image_path}")
        
        # Load and resize the image to save memory
        try:
            image = Image.open(image_path)
            
            # Resize large images to reduce memory usage
            width, height = image.size
            if width > 1000 or height > 1000:
                # Calculate new dimensions while maintaining aspect ratio
                if width > height:
                    new_width = 1000
                    new_height = int(height * (new_width / width))
                else:
                    new_height = 1000
                    new_width = int(width * (new_height / height))
                
                logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height} to save memory")
                image = image.resize((new_width, new_height))
        except Exception as img_err:
            logger.error(f"Error loading or resizing image: {img_err}")
            return f"[Error loading image: {str(img_err)}]"
        
        # Process with more memory-efficient settings
        try:
            # Use memory-efficient OCR configuration
            # OEM 1: Legacy Tesseract engine (less memory intensive)
            # PSM 6: Assume a single uniform block of text
            custom_config = r'--oem 1 --psm 6'
            
            # Process with handwritten mode if specified
            processed_image = preprocess_image(image, is_handwritten=True)
            
            # Extract text with timeout to prevent process hanging
            text = pytesseract.image_to_string(
                processed_image,
                config=custom_config,
                lang='eng',
                timeout=30
            )
            
            # Clean up resources immediately
            del processed_image
            
            if len(text.strip()) < 20:
                logger.info("Handwritten text detection yielded little text. Trying printed text processing.")
                # Try again with printed text mode
                processed_image = preprocess_image(image, is_handwritten=False)
                alt_text = pytesseract.image_to_string(
                    processed_image,
                    config=custom_config,
                    lang='eng',
                    timeout=30
                )
                
                # Use the better result
                if len(alt_text.strip()) > len(text.strip()):
                    logger.info("Printed text processing yielded better results.")
                    text = alt_text
                
                # Clean up
                del processed_image
            
            # Clean up the original image
            del image
            
            logger.debug(f"Extracted text from image: {len(text)} characters")
            return text
            
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
                        
                        # Preprocess the image with specified mode
                        processed_page = preprocess_image(img, is_handwritten=True)
                        
                        # Use a simpler OCR configuration to reduce memory usage
                        # PSM 6: Assume a single uniform block of text
                        # OEM 1: Use the legacy Tesseract engine which is more memory efficient
                        custom_config = r'--oem 1 --psm 6'
                        
                        # Extract text using PyTesseract with timeout to prevent hanging
                        text = pytesseract.image_to_string(
                            processed_page,
                            config=custom_config,
                            lang='eng',  # Explicitly use English language
                            timeout=30   # Set timeout to prevent process hanging
                        )
                        
                        full_text += f"\n--- Page {page_num} ---\n{text}\n"
                    except Exception as ocr_error:
                        logger.error(f"OCR error on page {page_num}: {ocr_error}")
                        full_text += f"\n--- Page {page_num} ---\n[OCR processing error]\n"
                    finally:
                        # Free memory immediately
                        del processed_page
                        del img
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
                        
                        # Clean the text of any problematic characters
                        page_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', page_text)
                        
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
                
                # Clean NUL bytes and other control characters
                extracted_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', extracted_text)
                
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
                    
                    # Clean the text further
                    # Remove control characters except newlines and tabs
                    text_content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text_content)
                    
                    # If we have meaningful content, return it
                    if len(text_content.strip()) > 200:
                        logger.info("Successfully extracted text using binary filtering")
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
                img = img.resize((new_width, new_height), Image.LANCZOS)
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
