# -*- coding: utf-8 -*-
from typing import Dict, List
from .constants import CONTENT_EXAMPLES, DIALOG_EXAMPLES

def find_similar_examples(message: str, examples: List[Dict], top_k: int = 3) -> List[Dict]:
    """Find most similar examples (word overlap) - RAG Logic from notebook"""
    message_words = set(message.lower().split())
    similarities = []
    for ex in examples:
        ex_words = set(ex["message"].lower().split())
        overlap = len(message_words & ex_words)
        similarities.append((overlap, ex))
    similarities.sort(key=lambda x: x[0], reverse=True)
    return [ex for _, ex in similarities[:top_k]]

def create_content_prompt(message: str, features: Dict) -> str:
    """Create prompt for content evaluation (0-3)"""
    similar_examples = find_similar_examples(message, CONTENT_EXAMPLES)
    
    examples_text = ""
    for i, ex in enumerate(similar_examples, 1):
        examples_text += f"\n\nÖrnek {i}:\nMesaj: \"{ex['message']}\"\nPuan: {ex['score']}\nGerekçe: {ex['reasoning']}"

    basic_str = ', '.join(features['basic_terms'][:5]) if features['basic_terms'] else "YOK"
    advanced_str = ', '.join(features['advanced_terms'][:5]) if features['advanced_terms'] else "YOK"

    return f"""Sen bir eğitim değerlendirme uzmanısın. Ortaokul öğrencilerinin iklim ve çevre konulu chatbot ile yaptığı konuşmaları değerlendiriyorsun.

İçerik Gelişmişliği Kriteri (0-3):
0 = Konu Dışı: Çevre, iklim, enerji, su veya tarım ile ilgili hiçbir terim yok.
1 = Temel Kavram: Sadece temel terimler var (örn: "su", "geri dönüşüm", "plastik"). İleri terim YOK.
2 = İleri Kavram: En az 1 ileri terim var (örn: "karbon ayak izi", "iklim değişikliği", "sürdürülebilirlik") AMA başka kavramla mantıksal bağlantı yok.
3 = Çoklu Kavram: En az 1 ileri terim VE en az 2 farklı kavram arası ilişki/akıl yürütme var.

{examples_text}

Mesajda Bulunan Özellikler:
- Temel terimler ({features['basic_term_count']}): {basic_str}
- İleri terimler ({features['advanced_term_count']}): {advanced_str}
- Akıl yürütme: {"VAR" if features['has_reasoning'] else "YOK"}
- İlişki belirteci: {"VAR" if features['has_connection'] else "YOK"}

ŞİMDİ DEĞERLENDİRİLECEK MESAJ:
"{message}"

CEVAP FORMATI (sadece JSON):
{{"score": <0, 1, 2 veya 3>, "reasoning": "<Türkçe gerekçe>"}}"""

def create_dialog_prompt(message: str, features: Dict) -> str:
    """Create prompt for dialog evaluation (A-D)"""
    similar_examples = find_similar_examples(message, DIALOG_EXAMPLES)
    
    examples_text = ""
    for i, ex in enumerate(similar_examples, 1):
        examples_text += f"\n\nÖrnek {i}:\nMesaj: \"{ex['message']}\"\nPuan: {ex['score']}\nGerekçe: {ex['reasoning']}"

    return f"""Sen bir eğitim değerlendirme uzmanısın. Ortaokul öğrencilerinin iklim ve çevre konulu chatbot ile yaptığı konuşmaları değerlendiriyorsun.

Diyalog Seviyesi Kriteri (A-D):
A = Minimal: Tek kelime, emoji, diyalogu ilerletmeyen, 2 kelime veya daha az.
B = Basit: Tanımlayıcı sorular ("X nedir?"), yüzeysel sorular, önceki mesaja atıf yok.
C = İlerletici: Referans var ("peki", "dediğin gibi"), öneri/fikir sunuyor, veya etkisel soru soruyor.
D = Akıl Yürütme: Güçlü akıl yürütme ("çünkü", "bu nedenle") veya detaylı eylem planı (20+ kelime).

{examples_text}

Mesaj Özellikleri:
- Kelime sayısı: {features['word_count']}
- Soru mu: {"EVET" if features['is_question'] else "HAYIR"}
- Akıl yürütme: {"EVET" if features['has_reasoning'] else "HAYIR"}
- Eylem/öneri: {"EVET" if features['has_action'] else "HAYIR"}

ŞİMDİ DEĞERLENDİRİLECEK MESAJ:
"{message}"

CEVAP FORMATI (sadece JSON):
{{"score": "<A, B, C veya D>", "reasoning": "<Türkçe gerekçe>"}}"""
