"""Translation Tool for Foundry agent registration

Provides a Foundry tool function to translate text between English and Spanish.
"""

from __future__ import annotations
from typing import Dict, Any
import logging
import json
import os

try:
    from deep_translator import GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False
    GoogleTranslator = None

logger = logging.getLogger(__name__)

# Import code tool registry
try:
    from .code_tools_registry import register_code_tool
except ImportError:
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.append(str(repo_root.parent))
    from src.news_reporter.tools.code_tools_registry import register_code_tool


def _detect_language(text: str) -> str:
    """Detect the language of the input text using deep-translator."""
    if not _TRANSLATOR_AVAILABLE:
        # Fallback: simple heuristic
        spanish_chars = set('áéíóúñüÁÉÍÓÚÑÜ¿¡')
        spanish_count = sum(1 for char in text if char in spanish_chars)
        return 'es' if spanish_count > 0 else 'en'
    
    try:
        detected = GoogleTranslator().detect(text)
        return detected.lower() if detected else 'en'
    except Exception:
        # Fallback to simple heuristic
        spanish_chars = set('áéíóúñüÁÉÍÓÚÑÜ¿¡')
        spanish_count = sum(1 for char in text if char in spanish_chars)
        return 'es' if spanish_count > 0 else 'en'


@register_code_tool
def translate_text(text: str, target_language: str = "auto", source_language: str = "auto") -> str:
    """
    Foundry tool function for translating text between English and Spanish.
    
    Translates text from English to Spanish or from Spanish to English.
    If target_language is "auto", it will automatically detect the source language
    and translate to the opposite language (English <-> Spanish).
    
    Args:
        text: The text to translate
        target_language: Target language code ("en" for English, "es" for Spanish, "auto" for auto-detect)
        source_language: Source language code ("en", "es", or "auto" for auto-detect)
    
    Returns:
        JSON string with translation result including:
        - original_text: The original input text
        - translated_text: The translated text
        - source_language: Detected/specified source language
        - target_language: Target language used
        - success: Whether translation succeeded
        - error: Error message if translation failed (optional)
    """
    try:
        if not _TRANSLATOR_AVAILABLE:
            # Fallback: simple word replacement if translator not available
            logger.warning("deep-translator not available, using fallback translation")
            detected = _detect_language(text)
            return json.dumps({
                "original_text": text,
                "translated_text": text,  # No translation available
                "source_language": detected,
                "target_language": "auto",
                "success": False,
                "error": "Translation library (deep-translator) not installed. Install with: pip install deep-translator"
            }, indent=2)
        
        # Handle auto-detect for target language
        if target_language == "auto":
            detected = _detect_language(text)
            if detected == "en":
                target_language = "es"
            else:
                target_language = "en"
        
        # Handle auto-detect for source language
        if source_language == "auto":
            source_language = _detect_language(text)
        
        # Normalize language codes for deep-translator
        lang_map = {"en": "en", "es": "es", "english": "en", "spanish": "es"}
        target_lang = lang_map.get(target_language.lower(), target_language.lower())
        source_lang = lang_map.get(source_language.lower(), source_language.lower()) if source_language != "auto" else None
        
        # Perform translation
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        translated_text = translator.translate(text)
        
        # Detect actual source language if auto was used
        if source_language == "auto":
            detected_source = _detect_language(text)
        else:
            detected_source = source_lang
        
        return json.dumps({
            "original_text": text,
            "translated_text": translated_text,
            "source_language": detected_source,
            "target_language": target_lang,
            "success": True,
        }, indent=2)
        
    except Exception as e:
        logger.exception(f"Error translating text: {e}")
        error_result = {
            "original_text": text,
            "translated_text": "",
            "source_language": source_language if source_language != "auto" else "unknown",
            "target_language": target_language if target_language != "auto" else "unknown",
            "success": False,
            "error": f"Translation failed: {str(e)}"
        }
        return json.dumps(error_result, indent=2)
