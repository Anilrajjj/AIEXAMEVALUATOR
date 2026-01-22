import os
import logging
import base64
import json
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import requests
except Exception:
    requests = None


def extract_text_with_gemini(file_path: str, is_handwritten: bool = True, timeout: int = 30) -> Optional[str]:
    """
    Attempt to extract text from a file using a Gemini-compatible API.

    Behavior:
    - Reads `GEMINI_API_KEY` from environment. If not set, returns None.
    - If `GEMINI_API_URL` is set in environment, will POST file bytes to that URL
      with an Authorization header. Otherwise will attempt a reasonable
      Google Generative Language endpoint using the API key as a query param.

    Note: This is a best-effort wrapper. The exact Gemini/GenAI HTTP API
    contract may differ; callers should treat failures gracefully.
    """
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        logger.debug("GEMINI_API_KEY not set; skipping Gemini extraction")
        return None

    if requests is None:
        logger.warning("requests library not available; cannot call Gemini API")
        return None

    # Read file bytes
    try:
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
    except Exception as e:
        logger.error(f"Unable to read file for Gemini extraction: {e}")
        return None

    # Build request
    api_url = os.environ.get('GEMINI_API_URL')
    headers = {
        'Accept': 'application/json',
    }

    payload = {
        'input': {
            'mime_type': 'application/octet-stream',
            'data_base64': base64.b64encode(file_bytes).decode('utf-8')
        },
        'features': {
            'extract_text': True,
            'handwritten': bool(is_handwritten)
        }
    }

    try:
        if api_url:
            # Use custom URL with Authorization header
            headers['Authorization'] = f"Bearer {api_key}"
            resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
        else:
            # Try Google Generative Language minimal endpoint pattern (best-effort)
            # Put the key in query param as many Google APIs accept this form
            url = f"https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generate?key={api_key}"
            headers['Content-Type'] = 'application/json'
            resp = requests.post(url, json={'prompt': payload}, headers=headers, timeout=timeout)

        if resp.status_code != 200:
            logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
            return None

        data = resp.json()

        # Attempt to extract text from common response shapes
        if isinstance(data, dict):
            # Common field used by many generative APIs
            text = None
            # Try a few possible fields
            text = data.get('text') or data.get('output') or data.get('response')
            if not text:
                # Some APIs return nested choices or generations
                if 'candidates' in data and isinstance(data['candidates'], list):
                    text = ' '.join([c.get('content', '') for c in data['candidates'] if isinstance(c, dict)])
                elif 'choices' in data and isinstance(data['choices'], list):
                    text = ' '.join([ch.get('text', '') or ch.get('message', {}).get('content', '') for ch in data['choices']])

            if text:
                return text

        # If we couldn't parse anything meaningful, return None
        logger.debug("Gemini API response did not contain extractable text")
        return None

    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return None
