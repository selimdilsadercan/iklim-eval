# -*- coding: utf-8 -*-
import pandas as pd
import re
from pathlib import Path
from typing import Tuple, Optional

class DataHandler:
    @staticmethod
    def load_student_csv(file_path: str) -> Tuple[pd.DataFrame, str]:
        """CSV yükler ve sütun adlarını standartlaştırır"""
        path = Path(file_path)
        student_name = path.stem.replace("_", " ") # Dosya adından isim alma fatma_nur -> fatma nur
        
        # Farklı encoding'leri dene
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(file_path, encoding='latin-1')

        # --- Sütun Standartlaştırma ---
        col_map = {}
        for col in df.columns:
            l_col = col.lower().strip()
            if l_col in ['sender', 'user', 'gönderen']:
                col_map[col] = 'Sender'
            elif l_col in ['message', 'mesaj', 'text']:
                col_map[col] = 'Message'
            elif l_col in ['student', 'öğrenci', 'ad']:
                col_map[col] = 'Student'
        
        df = df.rename(columns=col_map)

        # Eksik sütunları tamamla
        if 'Sender' not in df.columns:
            raise ValueError(f"CSV'de gönderen (User/Sender) sütunu bulunamadı: {file_path}")
        if 'Message' not in df.columns:
            raise ValueError(f"CSV'de mesaj sütunu bulunamadı: {file_path}")
        if 'Student' not in df.columns:
            df['Student'] = student_name # Dosya adını kullan

        # Gönderen değerlerini standartlaştır (Bot, User -> bot, student)
        df['Sender'] = df['Sender'].apply(lambda x: 'student' if str(x).lower() in ['user', 'öğrenci', 'student'] else 'bot')

        return df, student_name

    @staticmethod
    def parse_metadata_from_path(file_path: str) -> dict:
        """Klasör adından sınıf ve okul bilgisini çıkarır"""
        folder_name = Path(file_path).parent.name
        # Örnek: 'karaagac_ortaokulu.6C_tum_botlar...'
        match = re.search(r"([a-zA-ZçğıöşüÇĞİÖŞÜ_]+)\.([0-9A-Z]+)_", folder_name)
        if match:
            return {
                "school": match.group(1).replace("_", " ").title(),
                "class": match.group(2).upper()
            }
        return {"school": "Bilinmiyor", "class": "Bilinmiyor"}
