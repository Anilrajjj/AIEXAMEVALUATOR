
"""
OCR Configuration Helper

This file helps configure cloud-based OCR services for better handwritten text recognition.

To use Google Vision API:
1. Go to Google Cloud Console
2. Enable Vision API
3. Create service account and download JSON key
4. Set environment variable: export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

To use Azure Computer Vision:
1. Go to Azure Portal
2. Create Computer Vision resource
3. Set environment variables:
   - AZURE_VISION_ENDPOINT="https://your-resource.cognitiveservices.azure.com/"
   - AZURE_VISION_KEY="your-subscription-key"

Alternative: Use OCR.Space API (Free tier available)
1. Sign up at ocr.space
2. Set environment variable: OCR_SPACE_API_KEY="your-api-key"
"""

import os
import requests
import base64
import logging

logger = logging.getLogger(__name__)

def extract_text_with_ocr_space(image_path):
    """
    Extract text using OCR.Space API (free tier available)
    Better for handwritten text than Tesseract
    """
    try:
        api_key = os.getenv('OCR_SPACE_API_KEY')
        if not api_key:
            logger.warning("OCR.Space API key not found")
            return None
            
        logger.info("Using OCR.Space API for handwritten text recognition")
        
        # Prepare the image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
        
        # OCR.Space API endpoint
        url = 'https://api.ocr.space/parse/image'
        
        payload = {
            'apikey': api_key,
            'base64Image': f'data:image/jpeg;base64,{image_data}',
            'language': 'eng',
            'detectOrientation': 'true',
            'scale': 'true',
            'OCREngine': '2',  # Use engine 2 which is better for handwriting
            'isTable': 'false'
        }
        
        response = requests.post(url, data=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get('IsErroredOnProcessing', True):
                error_msg = result.get('ErrorMessage', ['Unknown error'])[0]
                logger.error(f"OCR.Space API error: {error_msg}")
                return None
            
            # Extract text from all parsed results
            extracted_text = ""
            for parsed_result in result.get('ParsedResults', []):
                text = parsed_result.get('ParsedText', '')
                if text:
                    extracted_text += text + '\n'
            
            if extracted_text.strip():
                logger.info(f"OCR.Space API extracted {len(extracted_text)} characters")
                return f"[OCR.Space API - Handwriting Enhanced]\n{extracted_text}"
            else:
                logger.warning("OCR.Space API returned empty result")
                return None
        else:
            logger.error(f"OCR.Space API request failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"OCR.Space API error: {e}")
        return None

def get_ocr_setup_instructions():
    """Return setup instructions for OCR APIs"""
    return """
    To improve handwritten text recognition, set up one of these cloud OCR services:
    
    1. OCR.Space (Easiest - Free tier available):
       - Sign up at https://ocr.space/ocrapi
       - Get your free API key
       - Set environment variable: OCR_SPACE_API_KEY="your-key"
    
    2. Google Vision API (Most accurate):
       - Enable Vision API in Google Cloud Console
       - Create service account and download JSON key
       - Set GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
    
    3. Azure Computer Vision:
       - Create Computer Vision resource in Azure Portal
       - Set AZURE_VISION_ENDPOINT and AZURE_VISION_KEY environment variables
    """
