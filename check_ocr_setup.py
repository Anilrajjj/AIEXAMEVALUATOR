
#!/usr/bin/env python3
"""
OCR Setup Checker

Run this script to check which OCR services are properly configured.
"""

import os
import sys

def check_google_vision():
    """Check if Google Vision API is set up"""
    try:
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if credentials_path and os.path.exists(credentials_path):
            from google.cloud import vision
            client = vision.ImageAnnotatorClient()
            return True, "Google Vision API is properly configured"
        else:
            return False, "GOOGLE_APPLICATION_CREDENTIALS not set or file not found"
    except ImportError:
        return False, "Google Cloud Vision library not installed (pip install google-cloud-vision)"
    except Exception as e:
        return False, f"Google Vision API error: {str(e)}"

def check_azure_vision():
    """Check if Azure Computer Vision is set up"""
    endpoint = os.getenv('AZURE_VISION_ENDPOINT')
    key = os.getenv('AZURE_VISION_KEY')
    
    if endpoint and key:
        return True, "Azure Computer Vision API is configured"
    else:
        missing = []
        if not endpoint:
            missing.append('AZURE_VISION_ENDPOINT')
        if not key:
            missing.append('AZURE_VISION_KEY')
        return False, f"Missing environment variables: {', '.join(missing)}"

def check_ocr_space():
    """Check if OCR.Space API is set up"""
    api_key = os.getenv('OCR_SPACE_API_KEY')
    if api_key:
        return True, "OCR.Space API key is configured"
    else:
        return False, "OCR_SPACE_API_KEY environment variable not set"

def main():
    print("=== OCR Services Setup Check ===\n")
    
    services = [
        ("Google Vision API", check_google_vision),
        ("Azure Computer Vision", check_azure_vision),
        ("OCR.Space API", check_ocr_space)
    ]
    
    configured_count = 0
    
    for service_name, check_func in services:
        is_configured, message = check_func()
        status = "✅ CONFIGURED" if is_configured else "❌ NOT CONFIGURED"
        print(f"{service_name}: {status}")
        print(f"   {message}\n")
        
        if is_configured:
            configured_count += 1
    
    if configured_count == 0:
        print("⚠️  No cloud OCR services are configured.")
        print("   The system will use Tesseract OCR, which has limited handwriting recognition.")
        print("\nTo improve handwritten text recognition, configure at least one cloud OCR service:")
        print("\n1. OCR.Space (Free tier available):")
        print("   export OCR_SPACE_API_KEY='your-api-key'")
        print("\n2. Google Vision API:")
        print("   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account.json'")
        print("\n3. Azure Computer Vision:")
        print("   export AZURE_VISION_ENDPOINT='https://your-resource.cognitiveservices.azure.com/'")
        print("   export AZURE_VISION_KEY='your-subscription-key'")
    else:
        print(f"✅ {configured_count} OCR service(s) configured for enhanced handwriting recognition!")

if __name__ == "__main__":
    main()
