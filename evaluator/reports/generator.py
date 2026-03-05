# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict

class ReportGenerator:
    def __init__(self, scorer_instance):
        self.scorer = scorer_instance
        self.dialog_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        self.num_to_dialog = {v: k for k, v in self.dialog_map.items()}

    def compute_statistics(self, df: pd.DataFrame) -> dict:
        """Etiketlenmiş veriden pedagojik istatistikleri hesaplar"""
        sdf = df[df["Sender"] == "student"].copy()
        if sdf.empty: return {}

        # Veri tiplerini zorla
        sdf["content_label"] = pd.to_numeric(sdf["content_label"], errors='coerce').fillna(0)
        sdf["dialog_num"] = sdf["dialog_label"].map(self.dialog_map).fillna(0)

        n = len(sdf)
        avg_content = sdf["content_label"].mean()
        avg_dialog_num = sdf["dialog_num"].mean()
        avg_dialog_cat = self.num_to_dialog[int(round(avg_dialog_num))]

        # Gelişim trendi
        if n >= 4:
            first_score = sdf.iloc[:2]["content_label"].mean()
            last_score  = sdf.iloc[-2:]["content_label"].mean()
        elif n >= 2:
            first_score = float(sdf.iloc[0]["content_label"])
            last_score  = float(sdf.iloc[-1]["content_label"])
        else:
            first_score = last_score = float(sdf.iloc[0]["content_label"])

        if last_score > first_score + 0.5: trend = "Yükselen (Öğrenme Gerçekleşiyor)"
        elif last_score < first_score - 0.5: trend = "Düşen (Odak Kaybı veya Zorlanma)"
        else: trend = "Stabil (Seviye Korunuyor)"

        # Dağılımlar
        content_dist = {str(k): int(v) for k, v in sorted(sdf["content_label"].value_counts().items())}
        dialog_dist  = {str(k): int(v) for k, v in sorted(sdf["dialog_label"].value_counts().items())}

        # Genel puan (1-10)
        content_norm = avg_content / 3.0
        dialog_norm  = avg_dialog_num / 3.0
        overall_score = max(1.0, min(10.0, round((content_norm * 0.6 + dialog_norm * 0.4) * 10, 1)))

        return {
            "overall_score": overall_score,
            "avg_content_score": round(avg_content, 2),
            "avg_dialog_level_category": avg_dialog_cat,
            "trend": trend,
            "content_score_distribution": content_dist,
            "dialog_level_distribution": dialog_dist,
            "message_count": n
        }

    def generate_final_report(self, df: pd.DataFrame, student_name: str, meta: dict) -> dict:
        """İstatistikleri ve LLM özetini birleştirir"""
        stats = self.compute_statistics(df)
        
        # Claude için özet prompt'u
        msg_summary = "\n".join([f"- {row['Message'][:100]} (C:{row['content_label']} D:{row['dialog_label']})" 
                                for _, row in df[df['Sender']=='student'].iterrows()])
        
        prompt = f"""Bir eğitim uzmanı olarak şu öğrenciyi değerlendir:
Öğrenci: {student_name}
İstatistikler: {stats}
Mesajlar:
{msg_summary}

Lütfen şu formatta bir JSON döndür:
{{
  "engagement_level": "<Düşük|Orta|Yüksek>",
  "conversation_themes": ["tema1", "tema2"],
  "summary_paragraph": "Türkçe 2-3 cümlelik pedagojik özet."
}}"""

        # Scorer içindeki LLM'i kullanarak özeti al
        try:
            llm_eval = self.scorer.score_with_llm(prompt, {}, score_type="summary") # placeholder features {}
        except:
            llm_eval = {"engagement_level": "Bilinmiyor", "summary_paragraph": "Özet oluşturulamadı."}

        return {
            "metadata": {
                "student_name": student_name,
                "school": meta.get("school"),
                "class": meta.get("class"),
                "generated_at": datetime.now().isoformat()
            },
            "stats": stats,
            "qualitative_analysis": llm_eval
        }
