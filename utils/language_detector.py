import re
from typing import Dict, Any

class LanguageDetector:
    """Simple language detector for English and Arabic"""
    
    def __init__(self):
        # Arabic Unicode ranges
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')
        # English pattern (basic Latin characters)
        self.english_pattern = re.compile(r'[a-zA-Z]')
    
    def detect_language(self, text: str) -> str:
        """
        Detect language based on character patterns
        Returns: 'English', 'Arabic', or 'Unknown'
        """
        if not text or not text.strip():
            return 'Unknown'
        
        # Count Arabic and English characters
        arabic_chars = len(self.arabic_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        
        # Calculate ratios
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return 'Unknown'
        
        arabic_ratio = arabic_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # Decision logic
        if arabic_ratio > 0.3:  # If more than 30% Arabic characters
            return 'Arabic'
        elif english_ratio > 0.5:  # If more than 50% English characters
            return 'English'
        elif arabic_ratio > english_ratio:
            return 'Arabic'
        elif english_ratio > arabic_ratio:
            return 'English'
        else:
            return 'Unknown'
    
    def get_language_stats(self, text: str) -> Dict[str, Any]:
        """Get detailed language statistics for the text"""
        arabic_chars = len(self.arabic_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        total_chars = len(text.replace(' ', ''))
        
        stats = {
            'arabic_characters': arabic_chars,
            'english_characters': english_chars,
            'total_characters': total_chars,
            'arabic_ratio': arabic_chars / max(total_chars, 1),
            'english_ratio': english_chars / max(total_chars, 1),
            'detected_language': self.detect_language(text)
        }
        
        return stats
    
    def is_arabic(self, text: str) -> bool:
        """Check if text is primarily Arabic"""
        return self.detect_language(text) == 'Arabic'
    
    def is_english(self, text: str) -> bool:
        """Check if text is primarily English"""
        return self.detect_language(text) == 'English'
    
    def get_recommended_model(self, text: str) -> str:
        """Get recommended model based on detected language"""
        language = self.detect_language(text)
        
        model_recommendations = {
            'Arabic': 'salti/bert-base-multilingual-cased-finetuned-squad',
            'English': 'bert-large-uncased-whole-word-masking-finetuned-squad',
            'Unknown': 'bert-base-multilingual-cased'
        }
        
        return model_recommendations.get(language, 'bert-base-multilingual-cased')
