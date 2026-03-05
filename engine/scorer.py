# -*- coding: utf-8 -*-
import os
import json
import re
import anthropic
from typing import Dict, Union
from .prompts import create_content_prompt, create_dialog_prompt

class Scorer:
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    @staticmethod
    def rule_based_content_score(features: Dict) -> int:
        """Notebook'taki kural tabanlı içerik puanlama mantığı"""
        total_terms = features["basic_term_count"] + features["advanced_term_count"]
        if total_terms == 0: return 0
        if features["advanced_term_count"] >= 1 and total_terms >= 2 and (features["has_reasoning"] or features["has_connection"]):
            return 3
        if features["advanced_term_count"] >= 1: return 2
        return 1

    @staticmethod
    def rule_based_dialog_score(features: Dict) -> str:
        """Notebook'taki kural tabanlı diyalog puanlama mantığı"""
        if features["is_minimal"]: return "A"
        if features["has_reasoning"] and not features["has_action"]: return "D"
        if features["has_reasoning"] and features["word_count"] > 20: return "D"
        if features["has_reference"] or features["has_action"]: return "C"
        return "B"

    def score_with_llm(self, message: str, features: Dict, score_type: str = "content") -> Dict:
        """Puanlama yapar: Önce LLM dener, hata alırsa kural tabanlıya döner"""
        rule_score = self.rule_based_content_score(features) if score_type == "content" else self.rule_based_dialog_score(features)
        
        if not self.client:
            return {"score": rule_score, "reasoning": "Sanal Ortam/Rule-based (API Key yok)", "method": "rule_based"}

        try:
            prompt = create_content_prompt(message, features) if score_type == "content" else create_dialog_prompt(message, features)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw_text = response.content[0].text.strip()
            # JSON temizleme
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                result["method"] = "llm"
                return result
            
        except Exception as e:
            print(f"  ⚠ LLM Error: {str(e)}")
            
        return {"score": rule_score, "reasoning": "Rule-based evaluation (LLM error or fallback)", "method": "rule_based"}
