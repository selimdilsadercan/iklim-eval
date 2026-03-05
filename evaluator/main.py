# -*- coding: utf-8 -*-
import os
import argparse
import json
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

from utils.data_handler import DataHandler
from engine.feature_extractor import FeatureExtractor
from engine.scorer import Scorer
from reports.generator import ReportGenerator

# .env dosyasını yükle
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="İklim-Eval: Öğrenci Mesaj Analiz Sistemi")
    parser.add_argument("csv_path", help="Analiz edilecek ham CSV dosyasının yolu")
    parser.add_argument("--output-dir", default="analysis_results", help="Sonuçların kaydedileceği klasör")
    args = parser.parse_args()

    # 1. Hazırlık
    print(f"🚀 Analiz başlatılıyor: {Path(args.csv_path).name}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Bileşenleri başlat
    extractor = FeatureExtractor() # Varsayılan terimlerle
    scorer = Scorer()
    reporter = ReportGenerator(scorer)

    # 2. Veri Yükleme ve Standartlaştırma
    try:
        df, student_name = DataHandler.load_student_csv(args.csv_path)
        meta = DataHandler.parse_metadata_from_path(args.csv_path)
        print(f"✅ Veri yüklendi. Öğrenci: {student_name} | Okul: {meta['school']}")
    except Exception as e:
        print(f"❌ Hata: {e}")
        return

    # 3. Adım Adım Etiketleme (Labeling)
    print("🧠 Mesajlar etiketleniyor...")
    student_indices = df[df['Sender'] == 'student'].index
    
    # Sonuç kolonlarını aç
    df['content_label'] = None
    df['dialog_label'] = None
    df['content_reasoning'] = None
    df['dialog_reasoning'] = None

    for idx in tqdm(student_indices):
        message = str(df.loc[idx, 'Message'])
        features = extractor.extract_features(message)
        
        # İçerik Puanlama
        c_res = scorer.score_with_llm(message, features, "content")
        df.at[idx, 'content_label'] = c_res['score']
        df.at[idx, 'content_reasoning'] = c_res['reasoning']
        
        # Diyalog Puanlama
        d_res = scorer.score_with_llm(message, features, "dialog")
        df.at[idx, 'dialog_label'] = d_res['score']
        df.at[idx, 'dialog_reasoning'] = d_res['reasoning']

    # 4. Rapor Oluşturma
    print("📊 Rapor hazırlanıyor...")
    report = reporter.generate_final_report(df, student_name, meta)

    # 5. Kaydetme
    safe_name = student_name.replace(" ", "_").lower()
    
    # Etiketli CSV'yi kaydet
    csv_out = os.path.join(args.output_dir, f"{safe_name}_labeled.csv")
    df.to_csv(csv_out, index=False, encoding='utf-8-sig')
    
    # JSON Raporu kaydet
    json_out = os.path.join(args.output_dir, f"{safe_name}_report.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✨ İşlem başarıyla tamamlandı!")
    print(f"📁 Etiketli Veri: {csv_out}")
    print(f"📄 Final Raporu: {json_out}")
    print(f"🏆 Genel Puan: {report['stats']['overall_score']}/10")

if __name__ == "__main__":
    main()
