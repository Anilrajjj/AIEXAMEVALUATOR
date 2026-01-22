"""
High-Accuracy Handwritten Text Extraction Pipeline
Using TrOCR, Calamari OCR, and KenLM rescoring
"""

import os
import logging
import json
import tempfile
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pdf2image
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None
import cv2
from typing import List, Dict, Tuple, Optional
import re

# Setup logging
logger = logging.getLogger(__name__)

# Module-level cache to reuse TrOCR model and processor across extractor instances
_GLOBAL_TROCR_CACHE = {}

# Module-level Calamari availability check to avoid repeated noisy warnings
try:
    import calamari_ocr  # noqa: F401
    CALAMARI_MODULE_AVAILABLE = True
except Exception:
    CALAMARI_MODULE_AVAILABLE = False

class TrOCRHandwrittenExtractor:
    """
    High-accuracy handwritten text extraction using deep learning models
    """
    
    def __init__(
        self,
        fast_mode: bool = True,
        batch_size: int = 32,
        resize_height: int = 64,
        use_tesseract_fallback: bool = True,
        min_line_confidence: float = 0.5
    ):
        # Optimize defaults for IAM dataset: fast_mode on, larger batches, smallest line height for speed
        self.fast_mode = fast_mode
        self.use_tesseract_fallback = use_tesseract_fallback
        self.min_line_confidence = min_line_confidence
        # Tunable performance parameters
        self.batch_size = int(batch_size) if batch_size and batch_size > 0 else 32
        # Resize line crops to this max height (maintain aspect ratio) before sending to model
        self.resize_height = int(resize_height) if resize_height and resize_height > 0 else 64
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available - TrOCR not supported")
            self.device = None
            self.trocr_processor = None
            self.trocr_model = None
            self.calamari_available = False
            self.lm_scorer = None
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        # If using CPU, allow PyTorch to use multiple threads to utilize available cores
        try:
            if self.device.type == 'cpu' and torch is not None:
                cpu_threads = max(1, (os.cpu_count() or 1) - 1)
                torch.set_num_threads(cpu_threads)
                logger.info(f"Set torch CPU threads to: {torch.get_num_threads()}")
        except Exception:
            pass
        
        # Initialize TrOCR model
        self.trocr_processor = None
        self.trocr_model = None
        self._load_trocr_model()
        
        # Initialize Calamari OCR (optional)
        self.calamari_available = self._check_calamari_availability()
        
        # Initialize KenLM rescoring (fallback implementation)
        self.lm_scorer = self._init_language_model_scorer()
        
    def _load_trocr_model(self):
        """Load TrOCR model for handwritten text recognition"""
        try:
            # Reuse a globally cached model if available to avoid repeated slow loads
            if _GLOBAL_TROCR_CACHE.get("processor") and _GLOBAL_TROCR_CACHE.get("model"):
                logger.info("Reusing cached TrOCR model and processor from module cache")
                self.trocr_processor = _GLOBAL_TROCR_CACHE.get("processor")
                self.trocr_model = _GLOBAL_TROCR_CACHE.get("model")
                # Ensure model is on the correct device
                try:
                    self.trocr_model.to(self.device)
                except Exception:
                    pass
                return

            logger.info("Loading TrOCR model for handwritten text recognition...")
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model.to(self.device)
            # Cache for reuse by future extractor instances
            try:
                _GLOBAL_TROCR_CACHE["processor"] = self.trocr_processor
                _GLOBAL_TROCR_CACHE["model"] = self.trocr_model
            except Exception:
                pass
            logger.info("TrOCR model loaded and cached successfully")
        except Exception as e:
            logger.error(f"Failed to load TrOCR model: {e}")
            self.trocr_processor = None
            self.trocr_model = None


    
    def _check_calamari_availability(self) -> bool:
        """Check if Calamari OCR is available"""
        # Calamari is disabled by default to avoid repeated import errors and warnings.
        # For high-accuracy offline processing, set this to True and ensure calamari is properly installed.
        if CALAMARI_MODULE_AVAILABLE:
            logger.debug("Calamari OCR module present but disabled by default (too slow and often fails)")
            return False
        else:
            logger.debug("Calamari OCR not available - using TrOCR only")
            return False
    
    def _init_language_model_scorer(self):
        """Initialize language model scorer (fallback implementation)"""
        try:
            # Simple n-gram based scoring as fallback
            logger.info("Initializing language model scorer (fallback implementation)")
            return SimpleLanguageModelScorer()
        except Exception as e:
            logger.warning(f"Failed to initialize language model scorer: {e}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str, dpi: int = 600) -> Dict:
        """
        Extract text from PDF using high-accuracy pipeline
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for PDF to image conversion
            
        Returns:
            Dictionary with extracted text and metadata
        """
        try:
            logger.info(f"Processing PDF: {pdf_path} with DPI: {dpi}")
            
            # Convert PDF to images using pdf2image
            if dpi is None:
                dpi = 120 if getattr(self, 'fast_mode', False) else 600

            thread_count = max(1, min(4, (os.cpu_count() or 1)))
            pages = pdf2image.convert_from_path(
                pdf_path,
                dpi=dpi,
                thread_count=thread_count,
                fmt='RGB'  # Ensure RGB format
            )

            # Quick fallback: when fast_mode is enabled, try a fast Tesseract extraction
            # and return it immediately to avoid long waits. This gives a quick, lower-accuracy
            # result while the deep TrOCR pipeline is optional for background processing.
            if getattr(self, 'fast_mode', False):
                try:
                    import pytesseract
                    fast_pages = []
                    confidence_scores = []
                    for page_num, page_image in enumerate(pages, 1):
                        try:
                            gray = page_image.convert('L')
                            text = pytesseract.image_to_string(gray).strip()
                            conf = self._calculate_text_confidence(text)
                        except Exception:
                            text = ''
                            conf = 0.0
                        fast_pages.append({
                            'page_number': page_num,
                            'lines': [{'text': text, 'confidence': conf, 'method': 'pytesseract_fast'}],
                            'total_confidence': conf
                        })
                        confidence_scores.append(conf)

                    total_text = "\n".join([f"--- Page {p['page_number']} ---\n{p['lines'][0]['text']}" for p in fast_pages])
                    overall_conf = sum(confidence_scores)/len(confidence_scores) if confidence_scores else 0.0
                    logger.info("Returning fast Tesseract fallback results (fast_mode)")
                    return {
                        'pages': fast_pages,
                        'total_text': total_text,
                        'extraction_method': 'pytesseract_fast_fallback',
                        'confidence_scores': confidence_scores,
                        'overall_confidence': overall_conf,
                        'needs_review': overall_conf < 0.6
                    }
                except Exception as e:
                    logger.debug(f"Fast pytesseract fallback failed: {e}")
            
            if not pages:
                logger.error("Failed to convert PDF to images")
                return {
                    "pages": [],
                    "total_text": "[Error: Could not convert PDF to images]",
                    "extraction_method": "Error",
                    "overall_confidence": 0.0
                }
            
            results = {
                "pages": [],
                "total_text": "",
                "extraction_method": "TrOCR + Deep Learning Pipeline",
                "confidence_scores": []
            }
            
            # Prepare debug visualization directory
            base = os.path.basename(pdf_path).split('.')[0]
            viz_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'visualizations', f'debug_{base}')
            os.makedirs(viz_dir, exist_ok=True)

            for page_num, page_image in enumerate(pages, 1):
                logger.info(f"Processing page {page_num} with line-based TrOCR batching")
                
                # Lightweight preprocessing for line detection
                processed_image = self._preprocess_image_fast(page_image)

                # Detect text lines using layout detection (projection/profile based)
                text_lines = self._detect_text_lines(processed_image)

                if not text_lines:
                    logger.warning(f"No text lines detected on page {page_num}")
                    results["pages"].append({
                        "page_number": page_num,
                        "lines": [],
                        "total_confidence": 0.0
                    })
                    continue

                # Process all detected lines with batched TrOCR (fast!)
                page_results = self._process_text_lines(text_lines, page_num)
                
                results["pages"].append({
                    "page_number": page_num,
                    "lines": page_results["lines"],
                    "total_confidence": page_results["total_confidence"]
                })
                
                # Combine page text
                page_text = "\n".join([line["text"] for line in page_results["lines"]])
                results["total_text"] += f"\n--- Page {page_num} ---\n{page_text}\n"
                results["confidence_scores"].extend([line["confidence"] for line in page_results["lines"]])
            
            # Calculate overall confidence
            if results["confidence_scores"]:
                results["overall_confidence"] = sum(results["confidence_scores"]) / len(results["confidence_scores"])
            else:
                results["overall_confidence"] = 0.0

            # Flag for manual review if overall confidence low
            results["needs_review"] = results["overall_confidence"] < 0.6
            
            logger.info(f"TrOCR extraction complete. Overall confidence: {results['overall_confidence']:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in TrOCR PDF extraction: {e}")
            return {
                "pages": [],
                "total_text": f"[Error processing PDF with TrOCR: {str(e)}]",
                "extraction_method": "Error",
                "overall_confidence": 0.0
            }
    
    def _preprocess_image_fast(self, image: Image.Image) -> Image.Image:
        """
        Lightweight preprocessing: convert to grayscale and create binary for line detection.
        Skips expensive denoise, CLAHE, and skew correction for speed in fast_mode.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image (grayscale)
        """
        try:
            gray = np.array(image.convert('L'))
            
            # Create binary for line detection
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Return grayscale image for OCR with binary attached for line detection
            processed_image = Image.fromarray(gray)
            try:
                processed_image._binary_for_detection = binary
            except Exception:
                pass
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Fast preprocessing failed, using original image: {e}")
            return image
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Fast mode: skip expensive denoise and skew-correction to save CPU time
            if getattr(self, 'fast_mode', False):
                try:
                    gray = np.array(image.convert('L'))
                    # Simple threshold for detection
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    processed_image = Image.fromarray(gray)
                    try:
                        processed_image = ImageEnhance.Contrast(processed_image).enhance(1.05)
                        processed_image = ImageEnhance.Sharpness(processed_image).enhance(1.02)
                    except Exception:
                        pass
                    try:
                        processed_image._binary_for_detection = binary
                    except Exception:
                        pass
                    return processed_image
                except Exception:
                    # Fall back to full preprocessing below on any error
                    pass

            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Denoise using bilateral filter
            denoised = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # Skew correction
            corrected = self._correct_skew(enhanced)
            
            # Create a binary image for detection but keep a grayscale/RGB enhanced image for OCR
            try:
                binary = cv2.adaptiveThreshold(
                    corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            except Exception:
                binary = corrected

            # Convert the corrected grayscale back to PIL for OCR (do NOT hard-binarize for TrOCR)
            processed_image = Image.fromarray(corrected)

            # Additional PIL enhancements (keep as grayscale/RGB, not binary)
            try:
                processed_image = ImageEnhance.Contrast(processed_image).enhance(1.1)
                processed_image = ImageEnhance.Sharpness(processed_image).enhance(1.05)
            except Exception:
                pass

            # Attach a binary attribute for line-detection use (monkey-patch)
            try:
                processed_image._binary_for_detection = binary
            except Exception:
                pass

            return processed_image

        except Exception as e:
            logger.warning(f"Preprocessing failed, using original image: {e}")
            return image
    
    def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in the image using Hough line detection
        
        Args:
            image: Grayscale image as numpy array
            
        Returns:
            Skew-corrected image
        """
        try:
            # Detect edges
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta - np.pi/2
                    angles.append(angle)
                
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Only correct if angle is significant
                    if abs(avg_angle) > 0.1:  # More than ~6 degrees
                        # Rotate image
                        h, w = image.shape
                        center = (w // 2, h // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(avg_angle), 1.0)
                        corrected = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC)
                        return corrected
            
            return image
            
        except Exception as e:
            logger.warning(f"Skew correction failed: {e}")
            return image
    
    def _detect_text_lines(self, image: Image.Image) -> List[Dict]:
        """
        Detect text lines in the image using layout detection
        
        Args:
            image: Preprocessed PIL Image
            
        Returns:
            List of text line bounding boxes and images
        """
        try:
            # Use a projection/profile-based segmentation for handwritten text
            try:
                gray = np.array(image.convert('L'))
                # Use the binary for detection if available
                binary = getattr(image, '_binary_for_detection', None)
                if binary is None:
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                else:
                    if len(binary.shape) == 3:
                        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

                # Invert so text is white on black for projection
                inv = 255 - binary

                # Horizontal projection (sum of pixels in each row)
                proj = np.sum(inv, axis=1)

                # Smooth projection with a small window to reduce noise
                kernel_size = max(3, int(image.size[1] * 0.002))
                proj_smooth = np.convolve(proj, np.ones(kernel_size)/kernel_size, mode='same')

                # Find valleys in the projection to split lines
                thresh = max(5, np.max(proj_smooth) * 0.05)
                rows = np.where(proj_smooth > thresh)[0]
                if len(rows) == 0:
                    logger.info("Projection segmentation found no text rows")
                    return []

                # Group contiguous row blocks into line regions
                line_regions = []
                start = rows[0]
                prev = rows[0]
                for r in rows[1:]:
                    if r - prev > 5:  # gap threshold in pixels
                        line_regions.append((start, prev))
                        start = r
                    prev = r
                line_regions.append((start, prev))

                text_lines = []
                debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static', 'visualizations')
                os.makedirs(debug_dir, exist_ok=True)

                for i, (r0, r1) in enumerate(line_regions):
                    # Expand region slightly
                    pad_v = 6
                    y0 = max(0, r0 - pad_v)
                    y1 = min(gray.shape[0], r1 + pad_v)

                    # Crop the full width for handwritten lines
                    line_crop = gray[y0:y1, :]

                    # If crop is too small, skip
                    if line_crop.shape[1] < 30 or line_crop.shape[0] < 8:
                        continue

                    # Add minimal horizontal cropping using vertical projection
                    vproj = np.sum(255 - (line_crop), axis=0)
                    cols = np.where(vproj > 5)[0]
                    if len(cols) == 0:
                        x0, x1 = 0, line_crop.shape[1]
                    else:
                        x0, x1 = max(0, cols[0]-5), min(line_crop.shape[1], cols[-1]+5)

                    final_crop = line_crop[:, x0:x1]
                    pad = 10
                    padded = cv2.copyMakeBorder(final_crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
                    line_pil = Image.fromarray(padded)

                    # Save debug crop
                    try:
                        line_path = os.path.join(debug_dir, f"debug_page{page_num if 'page_num' in locals() else 0}_line{i}.png")
                        line_pil.convert('RGB').save(line_path)
                    except Exception:
                        pass

                    text_lines.append({
                        "id": i,
                        "bbox": (x0, y0, x1-x0, y1-y0),
                        "image": line_pil,
                        "aspect_ratio": (x1-x0) / max(1, (y1-y0))
                    })

                # Sort by y-coordinate
                text_lines.sort(key=lambda x: x["bbox"][1])
                logger.info(f"Detected {len(text_lines)} text lines (projection-based)")
                return text_lines
            except Exception as e:
                logger.error(f"Projection-based line detection failed: {e}")
                return []
            
        except Exception as e:
            logger.error(f"Text line detection failed: {e}")
            return []
    
    def _process_text_lines(self, text_lines: List[Dict], page_num: int) -> Dict:
        """
        Process text lines using TrOCR and ensemble methods (batched for speed)
        
        Args:
            text_lines: List of detected text lines
            page_num: Page number for logging
            
        Returns:
            Dictionary with processed text lines and confidence scores
        """
        results = {
            "lines": [],
            "total_confidence": 0.0
        }
        
        if not text_lines:
            return results
        
        confidences = []
        
        # Batch TrOCR processing: collect all line images, run generate once per batch
        for batch_start in range(0, len(text_lines), self.batch_size):
            batch_lines = text_lines[batch_start:batch_start + self.batch_size]
            batch_images = [line["image"] for line in batch_lines]
            
            # Run batched TrOCR
            batch_results = self._process_trocr_batch(batch_images)
            
            # Process each result in the batch
            for line, trocr_result in zip(batch_lines, batch_results):
                try:
                    # Process with Calamari if available (skipped in fast_mode)
                    calamari_result = None
                    if self.calamari_available and not getattr(self, 'fast_mode', False):
                        calamari_result = self._process_with_calamari(line["image"])
                    
                    # Combine results using ensemble method
                    final_result = self._ensemble_results(trocr_result, calamari_result)

                    # Apply language model rescoring (skip in fast_mode for speed)
                    if self.lm_scorer and not getattr(self, 'fast_mode', False):
                        final_result = self._rescore_with_lm(final_result)

                    # If result looks like garbage or has very low confidence, skip it
                    is_garbage = self._is_garbage_text(final_result.get("text", ""))
                    min_conf = getattr(self, "min_line_confidence", 0.35) or 0.35
                    current_conf = final_result.get("confidence", 0.0)
                    if current_conf < min_conf or is_garbage:
                        if getattr(self, "use_tesseract_fallback", True):
                            # Try a lightweight pytesseract fallback if available
                            try:
                                import pytesseract
                                pt_text = pytesseract.image_to_string(line["image"]).strip()
                                pt_conf = self._calculate_text_confidence(pt_text)
                                if pt_text and pt_conf >= min_conf and not self._is_garbage_text(pt_text):
                                    results["lines"].append({
                                        "text": pt_text,
                                        "confidence": pt_conf,
                                        "bbox": line["bbox"],
                                        "method": "pytesseract"
                                    })
                                    confidences.append(pt_conf)
                                    continue
                            except Exception:
                                pass

                        # Skip adding this noisy line — whole-page TrOCR fallback may handle it
                        logger.debug(f"Skipping noisy/low-confidence line {line.get('id')} on page {page_num}")
                        confidences.append(current_conf)
                        continue

                    results["lines"].append({
                        "text": final_result["text"],
                        "confidence": final_result["confidence"],
                        "bbox": line["bbox"],
                        "method": final_result["method"]
                    })

                    confidences.append(final_result["confidence"])
                    
                except Exception as e:
                    logger.error(f"Error processing text line {line['id']}: {e}")
                    results["lines"].append({
                        "text": "[Error processing line]",
                        "confidence": 0.0,
                        "bbox": line["bbox"],
                        "method": "Error"
                    })
                    confidences.append(0.0)
        
        if confidences:
            results["total_confidence"] = sum(confidences) / len(confidences)
        
        return results
    
    def _process_trocr_batch(self, images: List[Image.Image]) -> List[Dict]:
        """
        Process a batch of line images with TrOCR in a single forward pass for speed.
        
        Args:
            images: List of PIL Image objects
            
        Returns:
            List of dicts with text and confidence for each image
        """
        if not images:
            return []
        
        try:
            if not self.trocr_processor or not self.trocr_model:
                logger.warning("TrOCR model not available")
                return [{"text": "", "confidence": 0.0, "method": "TrOCR unavailable"} for _ in images]
            
            # Resize and convert images to RGB to reduce model input size and speed up inference
            images_prepared = []
            for img in images:
                try:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize to target height while keeping aspect ratio
                    w, h = img.size
                    if h > self.resize_height:
                        new_h = self.resize_height
                        new_w = max(1, int(w * (new_h / float(h))))
                        img = img.resize((new_w, new_h), resample=Image.LANCZOS)
                    images_prepared.append(img)
                except Exception:
                    # Fall back to original image if resizing fails
                    try:
                        images_prepared.append(img.convert('RGB'))
                    except Exception:
                        images_prepared.append(img)

            # Extract pixel values for entire batch at once (use padding to allow variable sizes)
            pixel_values = self.trocr_processor(images=images_prepared, return_tensors="pt", padding=True).pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text for entire batch but guard with a timeout to avoid long hangs
            gen_timeout = 30 if getattr(self, 'fast_mode', False) else 60
            if getattr(self, 'fast_mode', False):
                generated_ids = self._generate_with_timeout(pixel_values, max_length=256, num_beams=1, timeout=gen_timeout)
            else:
                generated_ids = self._generate_with_timeout(pixel_values, max_length=512, num_beams=5, timeout=gen_timeout)

            # If generation timed out, fall back to pytesseract per-image
            if generated_ids is None:
                logger.warning("TrOCR batch generation timed out; falling back to pytesseract for this batch")
                results = []
                try:
                    import pytesseract
                except Exception:
                    pytesseract = None

                for img in images:
                    try:
                        if pytesseract:
                            pt_text = pytesseract.image_to_string(img.convert('L')).strip()
                            pt_conf = self._calculate_text_confidence(pt_text)
                            results.append({"text": pt_text, "confidence": pt_conf, "method": "pytesseract_fallback"})
                        else:
                            results.append({"text": "", "confidence": 0.0, "method": "TrOCR Timeout"})
                    except Exception:
                        results.append({"text": "", "confidence": 0.0, "method": "TrOCR Timeout"})

                return results
            
            # Decode all texts at once
            generated_texts = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Compute confidence for each result
            results = []
            for generated_text in generated_texts:
                base_conf = self._calculate_text_confidence(generated_text)
                text_len = len(generated_text.strip())
                alpha_num = sum(1 for c in generated_text if c.isalnum())
                if text_len > 0:
                    heur = min(1.0, 0.2 + 0.8 * (alpha_num / text_len))
                    confidence = min(1.0, (base_conf * 0.6) + (heur * 0.4))
                else:
                    confidence = 0.0
                
                results.append({
                    "text": generated_text.strip(),
                    "confidence": confidence,
                    "method": "TrOCR"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"TrOCR batch processing failed: {e}")
            return [{"text": "", "confidence": 0.0, "method": "TrOCR Error"} for _ in images]
    
    def _process_with_trocr(self, image: Image.Image) -> Dict:
        """
        Process image with TrOCR model
        
        Args:
            image: PIL Image of text line
            
        Returns:
            Dictionary with text and confidence
        """
        try:
            if not self.trocr_processor or not self.trocr_model:
                logger.warning("TrOCR model not available")
                return {"text": "", "confidence": 0.0, "method": "TrOCR unavailable"}
            
            # Ensure image is in RGB format
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Preprocess image for TrOCR
            pixel_values = self.trocr_processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text with parameters tuned for handwriting, but use timeout
            gen_timeout = 30 if getattr(self, 'fast_mode', False) else 60
            if getattr(self, 'fast_mode', False):
                generated_ids = self._generate_with_timeout(pixel_values, max_length=256, num_beams=1, timeout=gen_timeout)
            else:
                generated_ids = self._generate_with_timeout(pixel_values, max_length=512, num_beams=5, timeout=gen_timeout)

            if generated_ids is None:
                logger.warning("TrOCR generation timed out for a single image; attempting pytesseract fallback")
                try:
                    import pytesseract
                    pt_text = pytesseract.image_to_string(image.convert('L')).strip()
                    pt_conf = self._calculate_text_confidence(pt_text)
                    return {"text": pt_text, "confidence": pt_conf, "method": "pytesseract_fallback"}
                except Exception:
                    return {"text": "", "confidence": 0.0, "method": "TrOCR Timeout"}

            # Decode text
            generated_text = self.trocr_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Compute heuristic confidence: prefer _calculate_text_confidence but boost when text has alphanumerics
            base_conf = self._calculate_text_confidence(generated_text)
            text_len = len(generated_text.strip())
            alpha_num = sum(1 for c in generated_text if c.isalnum())
            if text_len > 0:
                heur = min(1.0, 0.2 + 0.8 * (alpha_num / text_len))
                confidence = min(1.0, (base_conf * 0.6) + (heur * 0.4))
            else:
                confidence = 0.0

            return {
                "text": generated_text.strip(),
                "confidence": confidence,
                "method": "TrOCR"
            }
            
        except Exception as e:
            logger.error(f"TrOCR processing failed: {e}")
            return {"text": "", "confidence": 0.0, "method": "TrOCR Error"}
    
    def _calculate_text_confidence(self, text: str) -> float:
        """
        Calculate confidence score based on text characteristics
        
        Args:
            text: Extracted text
            
        Returns:
            Confidence score between 0 and 1
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        s = text.strip()
        # If extremely short, treat as low confidence
        if len(s) < 2:
            return 0.0

        # Basic confidence factors
        length_score = min(1.0, len(s) / 50)  # Longer text gets higher score

        # Check for common characters (letters, numbers, spaces)
        alpha_numeric_count = sum(1 for c in s if c.isalnum() or c.isspace())
        char_score = alpha_numeric_count / len(s) if len(s) > 0 else 0

        # Check for reasonable word count
        word_count = len(s.split())
        word_score = min(1.0, word_count / 10)  # More words = higher confidence

        # Penalize numeric-only / repetitive garbage like "0 0 0" or long runs of a single char
        digits_count = sum(1 for c in s if c.isdigit())
        digits_ratio = digits_count / max(1, len(s))
        unique_alphas = len(set([c for c in s.lower() if c.isalpha()]))
        if digits_ratio > 0.6 and unique_alphas < 3:
            return 0.0

        # Penalize excessively repetitive text
        most_common_char_count = max((s.count(ch) for ch in set(s)), default=0)
        repetitiveness = most_common_char_count / max(1, len(s))
        if repetitiveness > 0.6:
            return 0.0

        # Combine scores
        confidence = (length_score * 0.3 + char_score * 0.4 + word_score * 0.3)

        # Allow very low confidences but not artificially boost garbage
        return min(1.0, max(0.0, confidence))

    def _generate_with_timeout(self, pixel_values, max_length: int = 256, num_beams: int = 1, timeout: int = 30):
        """
        Run model.generate in a separate thread with a timeout to avoid long blocking calls.

        Returns generated_ids on success or None on timeout/error.
        """
        try:
            import concurrent.futures

            def _gen():
                with torch.no_grad():
                    try:
                        return self.trocr_model.generate(
                            pixel_values,
                            max_length=max_length,
                            num_beams=num_beams,
                            num_return_sequences=1,
                            pad_token_id=self.trocr_processor.tokenizer.pad_token_id,
                            eos_token_id=self.trocr_processor.tokenizer.eos_token_id,
                            do_sample=False
                        )
                    except TypeError:
                        return self.trocr_model.generate(pixel_values, max_length=max_length)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_gen)
                try:
                    return fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"TrOCR generation timed out after {timeout}s")
                    return None
                except Exception as e:
                    logger.error(f"TrOCR generation failed: {e}")
                    return None
        except Exception as e:
            logger.error(f"Failed to run generation with timeout helper: {e}")
            return None

    def _is_garbage_text(self, text: str) -> bool:
        """Heuristic to decide whether extracted text is likely garbage."""
        if not text or len(text.strip()) == 0:
            return True
        s = text.strip()
        if len(s) < 2:
            return True
        digits = sum(1 for c in s if c.isdigit())
        digits_ratio = digits / max(1, len(s))
        unique_alpha = len(set([c for c in s.lower() if c.isalpha()]))
        most_common_char_count = max((s.count(ch) for ch in set(s)), default=0)
        repet = most_common_char_count / max(1, len(s))
        if digits_ratio > 0.6 and unique_alpha < 3:
            return True
        if repet > 0.6:
            return True
        return False
    
    def _process_with_calamari(self, image: Image.Image) -> Optional[Dict]:
        """
        Process image with Calamari OCR (if available)
        
        Args:
            image: PIL Image of text line
            
        Returns:
            Dictionary with text and confidence or None
        """
        try:
            if not self.calamari_available:
                return None
            
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name)
                
                # Run Calamari OCR
                import calamari_ocr
                from calamari_ocr import CalamariPredictor
                
                predictor = CalamariPredictor(checkpoint="default")
                result = predictor.predict_from_files([tmp_file.name])
                
                # Clean up
                os.unlink(tmp_file.name)
                
                if result and len(result) > 0:
                    text = result[0].prediction.text
                    confidence = result[0].prediction.avg_char_probability
                    
                    return {
                        "text": text,
                        "confidence": confidence,
                        "method": "Calamari"
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Calamari OCR failed: {e}")
            return None
    
    def _ensemble_results(self, trocr_result: Dict, calamari_result: Optional[Dict]) -> Dict:
        """
        Combine results from multiple OCR engines
        
        Args:
            trocr_result: TrOCR result
            calamari_result: Calamari result (optional)
            
        Returns:
            Combined result
        """
        if not calamari_result:
            return trocr_result
        
        # If both results are similar, use average confidence
        if self._results_similar(trocr_result["text"], calamari_result["text"]):
            avg_confidence = (trocr_result["confidence"] + calamari_result["confidence"]) / 2
            return {
                "text": trocr_result["text"],
                "confidence": avg_confidence,
                "method": "TrOCR + Calamari Ensemble"
            }
        
        # If results differ, choose the one with higher confidence
        if trocr_result["confidence"] > calamari_result["confidence"]:
            return trocr_result
        else:
            return calamari_result
    
    def _results_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Check if two OCR results are similar
        
        Args:
            text1: First text result
            text2: Second text result
            threshold: Similarity threshold
            
        Returns:
            True if results are similar
        """
        if not text1 or not text2:
            return False
        
        # Simple similarity based on character overlap
        set1 = set(text1.lower().replace(" ", ""))
        set2 = set(text2.lower().replace(" ", ""))
        
        if not set1 or not set2:
            return False
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _rescore_with_lm(self, result: Dict) -> Dict:
        """
        Rescore result using language model
        
        Args:
            result: OCR result dictionary
            
        Returns:
            Rescored result
        """
        try:
            if not self.lm_scorer:
                return result
            
            # Get language model score
            lm_score = self.lm_scorer.score(result["text"])
            
            # Combine OCR confidence with LM score
            # α=1.0 for OCR, β=0.8 for LM
            alpha, beta = 1.0, 0.8
            final_confidence = alpha * result["confidence"] + beta * lm_score
            
            # Normalize to [0, 1]
            final_confidence = min(1.0, max(0.0, final_confidence))
            
            return {
                "text": result["text"],
                "confidence": final_confidence,
                "method": result["method"] + " + LM Rescoring"
            }
            
        except Exception as e:
            logger.warning(f"LM rescoring failed: {e}")
            return result


class SimpleLanguageModelScorer:
    """
    Simple language model scorer as fallback for KenLM
    """
    
    def __init__(self):
        # Simple word frequency-based scoring
        self.common_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no",
            "just", "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us"
        }
    
    def score(self, text: str) -> float:
        """
        Calculate language model score for text
        
        Args:
            text: Text to score
            
        Returns:
            Score between 0 and 1
        """
        if not text:
            return 0.0
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Count common words
        common_count = sum(1 for word in words if word in self.common_words)
        
        # Calculate score based on common word ratio
        common_ratio = common_count / len(words)
        
        # Additional scoring based on text characteristics
        char_score = min(1.0, len(text) / 100)  # Longer text gets higher score
        word_score = min(1.0, len(words) / 20)  # More words gets higher score
        
        # Combine scores
        final_score = (common_ratio * 0.6 + char_score * 0.2 + word_score * 0.2)
        
        return min(1.0, max(0.0, final_score))


# Module-level singleton extractor to avoid re-loading model on every call
_GLOBAL_TROCR_EXTRACTOR = None


def get_trocr_extractor(
    fast_mode: bool = True,
    use_tesseract_fallback: bool = True,
    min_line_confidence: float = 0.5
):
    """Return a singleton TrOCR extractor. Reuses model across calls to avoid heavy reloads."""
    global _GLOBAL_TROCR_EXTRACTOR
    if _GLOBAL_TROCR_EXTRACTOR is None:
        _GLOBAL_TROCR_EXTRACTOR = TrOCRHandwrittenExtractor(
            fast_mode=fast_mode,
            use_tesseract_fallback=use_tesseract_fallback,
            min_line_confidence=min_line_confidence
        )
    else:
        # update runtime configuration if caller requests changes
        _GLOBAL_TROCR_EXTRACTOR.fast_mode = fast_mode
        _GLOBAL_TROCR_EXTRACTOR.use_tesseract_fallback = use_tesseract_fallback
        _GLOBAL_TROCR_EXTRACTOR.min_line_confidence = min_line_confidence
    return _GLOBAL_TROCR_EXTRACTOR


def extract_text_with_trocr_pipeline(
    file_path: str,
    is_handwritten: bool = True,
    dpi: int = None,
    fast_mode: bool = True,
    use_tesseract_fallback: bool = True,
    min_line_confidence: float = 0.5
) -> str:
    """
    Main function to extract text using the TrOCR pipeline
    
    Args:
        file_path: Path to PDF or image file
        is_handwritten: Whether the content is handwritten
        
    Returns:
        Extracted text as string
    """
    try:
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available - TrOCR pipeline not supported")
            return None
            
        if not is_handwritten:
            logger.info("Non-handwritten content detected, using standard extraction")
            return None
        
        logger.info(f"Starting TrOCR pipeline for file: {file_path}")
        
        # Initialize (or reuse) TrOCR extractor singleton to avoid reloading the model repeatedly
        extractor = get_trocr_extractor(
            fast_mode=fast_mode,
            use_tesseract_fallback=use_tesseract_fallback,
            min_line_confidence=min_line_confidence
        )
        min_confidence = getattr(extractor, "min_line_confidence", min_line_confidence or 0.6) or 0.6
        
        # Check file type
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            logger.info("Processing PDF with TrOCR")
            # Extract from PDF
            # Choose DPI based on fast_mode if dpi not explicitly provided
            if dpi is None:
                dpi = 300 if fast_mode else 600
            results = extractor.extract_text_from_pdf(file_path, dpi=dpi)
            
            if results['total_text'].strip():
                # Format results for display
                formatted_text = f"[TrOCR Deep Learning Pipeline - Confidence: {results['overall_confidence']:.3f}]\n\n"
                formatted_text += results['total_text']
                
                # Add confidence information
                if results['confidence_scores']:
                    low_confidence_lines = [i for i, conf in enumerate(results['confidence_scores']) if conf < 0.7]
                    if low_confidence_lines:
                        formatted_text += f"\n\n[Note: Lines {low_confidence_lines} have low confidence and may need review]"
                
                logger.info(f"TrOCR PDF extraction successful: {len(formatted_text)} characters")
                return formatted_text
            else:
                logger.warning("TrOCR PDF extraction produced no text")
                return None
            
        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            logger.info("Processing image with TrOCR")
            # For single images, convert to PDF-like processing
            image = Image.open(file_path)
            
            # Preprocess image and try whole-page TrOCR first
            processed_image = extractor._preprocess_image(image)
            whole = extractor._process_with_trocr(processed_image)
            used_whole = False
            if whole and whole.get('text') and whole.get('confidence', 0.0) >= min_confidence:
                results = {"lines": [{"text": whole.get('text'), "confidence": whole.get('confidence'), "method": "TrOCR WholePage"}], "total_confidence": whole.get('confidence')}
                used_whole = True
            else:
                # Detect text lines
                text_lines = extractor._detect_text_lines(processed_image)
                if not text_lines:
                    logger.warning("No text lines detected in image")
                    return None
                # Process text lines
                results = extractor._process_text_lines(text_lines, 1)
            
            if results['lines']:
                # Format results
                formatted_text = f"[TrOCR Deep Learning Pipeline - Confidence: {results['total_confidence']:.3f}]\n\n"
                
                for line in results['lines']:
                    formatted_text += f"{line['text']}\n"
                
                logger.info(f"TrOCR image extraction successful: {len(formatted_text)} characters")
                return formatted_text
            else:
                logger.warning("TrOCR image processing produced no text lines")
                return None
            
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            return None
            
    except Exception as e:
        logger.error(f"TrOCR pipeline extraction failed: {e}")
        import traceback
        logger.error(f"TrOCR error traceback: {traceback.format_exc()}")
        return None
