"""
Hinglish (Hindi-English code-mixed) text preprocessing.

Handles:
- Romanized Hindi normalization (multiple spelling variations → standard form)
- Common Hinglish slang expansion
- Code-mixing detection
- Emoji-to-sentiment mapping
"""

import re
from typing import Dict, List, Tuple


class HinglishPreprocessor:
    """
    Preprocessor for Hinglish (Hindi-English code-mixed) text.
    
    Hinglish text is highly inconsistent — the same Hindi word can be
    romanized in many ways (e.g., "shukriya", "shukria", "sukriya").
    This class normalizes common variations.
    """
    
    def __init__(self):
        # Common Hinglish slang → normalized English meaning
        self.slang_dict = {
            # Positive expressions
            'accha': 'good', 'acha': 'good', 'achha': 'good',
            'badiya': 'great', 'badhiya': 'great', 'badia': 'great',
            'mast': 'awesome', 'mazza': 'fun', 'mazaa': 'fun',
            'zabardast': 'fantastic', 'kamaal': 'amazing',
            'shandaar': 'wonderful', 'shandar': 'wonderful',
            'jhakkas': 'superb', 'jhakaas': 'superb',
            'bindaas': 'carefree cool',
            
            # Negative expressions
            'bakwas': 'nonsense', 'bakwaas': 'nonsense',
            'ghatiya': 'terrible', 'ghatia': 'terrible',
            'bekar': 'useless', 'bekaar': 'useless',
            'wahiyat': 'awful', 'vahiyat': 'awful',
            'faltu': 'useless', 'faaltu': 'useless',
            'chutiya': 'stupid', 'chu': 'stupid',
            'gadha': 'donkey fool', 'gadhe': 'donkey fool',
            'kameena': 'mean person', 'kameene': 'mean person',
            'harami': 'scoundrel',
            
            # Sarcasm indicators
            'wah': 'wow sarcastic', 'waah': 'wow sarcastic',
            'kya baat': 'wow impressive',
            'sahi': 'right correct sarcastic', 'sahii': 'right correct sarcastic',
            
            # Common words
            'yaar': 'friend', 'yar': 'friend',
            'bhai': 'brother friend', 'bro': 'brother friend',
            'dost': 'friend', 'dosti': 'friendship',
            'paisa': 'money', 'paise': 'money',
            'kaam': 'work', 'naukri': 'job',
            'zindagi': 'life', 'jindagi': 'life',
            'pyaar': 'love', 'pyar': 'love', 'mohabbat': 'love',
            'dil': 'heart',
            'sapna': 'dream', 'sapne': 'dreams',
            
            # Intensifiers and fillers
            'bohot': 'very', 'bahut': 'very', 'boht': 'very',
            'ekdum': 'totally', 'bilkul': 'absolutely',
            'thoda': 'little', 'thodi': 'little',
            'zyada': 'more too much', 'jyada': 'more too much',
            'kuch': 'some', 'kuchh': 'some',
            'aur': 'and more', 'lekin': 'but',
            
            # Negation (important for sentiment!)
            'nahi': 'not', 'nhi': 'not', 'nai': 'not', 'na': 'not',
            'mat': 'do not', 'kabhi nahi': 'never',
            
            # Questions/Exclamations
            'kya': 'what', 'kaise': 'how', 'kyun': 'why', 'kyu': 'why',
            'kaha': 'where', 'kab': 'when', 'kaun': 'who',
            'hai': 'is', 'hain': 'are', 'tha': 'was', 'thi': 'was',
            
            # Internet slang
            'lol': 'laughing', 'lmao': 'laughing hard', 'rofl': 'laughing very hard',
            'smh': 'shaking head disappointed', 'bruh': 'brother disbelief',
            'fomo': 'fear of missing out', 'tbh': 'to be honest',
            'imo': 'in my opinion', 'ikr': 'i know right',
        }
        
        # Spelling variation patterns (regex)
        self.normalization_patterns = [
            # Repeated characters: "sooooo" → "so"
            (r'(.)\1{2,}', r'\1\1'),
            # "aa" variations: "bahut" vs "baahut"
            (r'aa+', 'a'),
            # "ee" variations
            (r'ee+', 'i'),
            # "oo" variations
            (r'oo+', 'u'),
        ]
    
    def normalize(self, text: str) -> str:
        """
        Full Hinglish normalization pipeline.
        
        1. Expand slang terms
        2. Normalize spelling variations
        3. Handle repeated characters
        """
        if not text:
            return ""
        
        words = text.lower().split()
        normalized = []
        
        for word in words:
            # Check slang dictionary first
            if word in self.slang_dict:
                normalized.append(self.slang_dict[word])
            else:
                # Apply normalization patterns
                norm_word = word
                for pattern, replacement in self.normalization_patterns:
                    norm_word = re.sub(pattern, replacement, norm_word)
                
                # Check slang dict again after normalization
                if norm_word in self.slang_dict:
                    normalized.append(self.slang_dict[norm_word])
                else:
                    normalized.append(norm_word)
        
        return " ".join(normalized)
    
    def detect_code_mixing(self, text: str) -> Dict:
        """
        Detect the degree of code-mixing in text.
        
        Returns dict with:
            is_mixed: bool — whether text contains both Hindi and English
            hindi_ratio: float — fraction of Hindi-origin words
            english_ratio: float — fraction of English words
        """
        if not text:
            return {'is_mixed': False, 'hindi_ratio': 0, 'english_ratio': 0}
        
        words = text.lower().split()
        hindi_words = sum(1 for w in words if w in self.slang_dict)
        total = len(words) if words else 1
        
        hindi_ratio = hindi_words / total
        english_ratio = 1 - hindi_ratio
        
        return {
            'is_mixed': hindi_ratio > 0.1 and english_ratio > 0.1,
            'hindi_ratio': hindi_ratio,
            'english_ratio': english_ratio,
        }
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing: clean + normalize Hinglish text."""
        if not text:
            return ""
        text = text.lower().strip()
        text = self.normalize(text)
        return text
