# -*- coding: utf-8 -*-
import re
import json
from typing import Dict, List, Tuple
from .constants import (
    REASONING_INDICATORS, WEAK_REASONING, STRONG_ACTION, 
    CONNECTION_INDICATORS, QUESTION_WORDS, MINIMAL_INDICATORS,
    REFERENCE_INDICATORS, CAUSAL_EFFECTUAL_INDICATORS, 
    ACTION_INDICATORS, MINIMAL_RESPONSES
)

class FeatureExtractor:
    def __init__(self, basic_terms: List[str] = None, advanced_terms: List[str] = None):
        # Varsayılan terim listeleri (JSON dosyası bulunamazsa kullanılacak)
        self.basic_terms = basic_terms or ["su", "plastik", "geri dönüşüm", "çöp", "orman", "deniz", "hava"]
        self.advanced_terms = advanced_terms or ["karbon ayak izi", "iklim değişikliği", "sürdürülebilirlik", "yenilenebilir enerji", "sera gazı", "fosil yakıt"]
        
    @staticmethod
    def get_term_variants(term: str) -> List[str]:
        """Generate possible variants of a term (Turkish morphology)"""
        variants = [term]
        if ' ' in term:
            words = term.split()
            variants.extend(words)
            for word in words:
                if len(word) > 4:
                    variants.extend([
                        word[:-1], word + 'lu', word + 'lı', 
                        word + 'sı', word + 'si', word + 'nı', word + 'ı'
                    ])
        return variants

    def extract_features(self, message: str) -> Dict:
        """Extract features from message - v2.1.1 Enhanced Logic"""
        message_lower = message.lower()
        
        # Term detection
        basic_found = []
        for term in self.basic_terms:
            for variant in self.get_term_variants(term):
                if len(variant) > 3 and re.search(r'\b' + re.escape(variant) + r'\w*\b', message_lower):
                    basic_found.append(term)
                    break

        advanced_found = []
        for term in self.advanced_terms:
            for variant in self.get_term_variants(term):
                if len(variant) > 3 and re.search(r'\b' + re.escape(variant) + r'\w*\b', message_lower):
                    advanced_found.append(term)
                    break

        # Special mappings (v2.1.1 notebook logic)
        if 'hidrolik' in message_lower or 'su enerjisi' in message_lower:
            if 'hidroelektrik enerji' not in advanced_found:
                advanced_found.append('hidroelektrik enerji')
        
        if 'tasarruflu' in message_lower and 'enerji' in message_lower:
            if 'enerji tasarrufu' not in advanced_found:
                advanced_found.append('enerji tasarrufu')

        # Logic checks
        has_reasoning = any(ind in message_lower for ind in REASONING_INDICATORS)
        has_weak_reasoning = any(wr in message_lower for wr in WEAK_REASONING)
        has_strong_action = any(sa in message_lower for sa in STRONG_ACTION)

        # Bugfix from notebook: Weak reasoning + strong action = no reasoning (Dialog C not D)
        if has_weak_reasoning and has_strong_action:
            has_reasoning = False

        word_count = len(message.split())
        is_very_short = any(mi in message_lower for mi in MINIMAL_INDICATORS) or word_count <= 2
        is_minimal = message_lower.strip() in MINIMAL_RESPONSES or is_very_short

        return {
            "basic_terms": list(set(basic_found)),
            "basic_term_count": len(set(basic_found)),
            "advanced_terms": list(set(advanced_found)),
            "advanced_term_count": len(set(advanced_found)),
            "has_reasoning": has_reasoning,
            "has_connection": any(conn in message_lower for conn in CONNECTION_INDICATORS),
            "is_question": message.strip().endswith("?") or any(qw in message_lower for qw in QUESTION_WORDS),
            "has_reference": any(ref in message_lower for ref in REFERENCE_INDICATORS),
            "has_causal_and_effectual": any(cef in message_lower for cef in CAUSAL_EFFECTUAL_INDICATORS),
            "has_action": any(act in message_lower for act in ACTION_INDICATORS),
            "word_count": word_count,
            "is_minimal": is_minimal
        }
