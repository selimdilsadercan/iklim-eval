# -*- coding: utf-8 -*-

# --- CONTENT SCORING EXAMPLES (Few-shot) ---
CONTENT_EXAMPLES = [
    {"message": "naber", "score": 0, "reasoning": "Çevre/iklim ile ilgisi yok, sosyal selamlaşma. Konu dışı."},
    {"message": "Geri dönüşüm nedir?", "score": 1, "reasoning": "Sadece 'geri dönüşüm' (temel terim) var. İleri terim yok."},
    {"message": "Caretta carettalar nerede yaşar?", "score": 1, "reasoning": "'Caretta caretta' (temel biyolojik terim) ama başka kavram yok."},
    {"message": "Karbon ayak izi nedir?", "score": 2, "reasoning": "'Karbon ayak izi' ileri terimdir ama başka kavram yok."},
    {"message": "iklim kanununu uygulanabılmesı için nasıl bilinçlendirme işlemleri yapılabilir", "score": 2, "reasoning": "İleri terim var ama başka kavram yok."},
    {"message": "Yenilenebilir enerji kullanırsak karbon ayak izimiz azalır mı", "score": 3, "reasoning": "'Yenilenebilir enerji' ve 'karbon ayak izi' olmak üzere iki ayrı kavramı ilişkilendirmiş."},
    {"message": "Geri dönüştürülebilir enerjiler yerine kömür kullanırsak hava nasıl etkilenir", "score": 3, "reasoning": "'Kömür' (fosil yakıt/ileri), geri dönüştürülebilir enerjiler ile hava arasında bağlantı kurmuş."},
    {"message": "Fosil yakıt kullanımı sera gazı emisyonunu arttırırsa iklim değişikliğine yol açar mı", "score": 3, "reasoning": "Üç ileri terim kullanılmış."}
]

# --- DIALOG SCORING EXAMPLES (Few-shot) ---
DIALOG_EXAMPLES = [
    {"message": "tamam", "score": "A", "reasoning": "Tek kelime, diyalogu ilerletmiyor."},
    {"message": "anladım 👍", "score": "A", "reasoning": "Kısa onay + emoji, soru yok, akıl yürütme yok."},
    {"message": "Geri dönüşüm nedir?", "score": "B", "reasoning": "Basit tanımlayıcı soru, önceki mesaja atıf yok."},
    {"message": "Karbon ayak izi ne demek?", "score": "B", "reasoning": "Tanımlayıcı soru, yüzeysel bilgi arayışı."},
    {"message": "Plastik poşeti dönüştürmezsek ne yapabiliriz?", "score": "C", "reasoning": "Alternatif çözümler soruyor, diyalogu ilerletiyor ama akıl yürütme yok."},
    {"message": "Peki karbonayak izinin hayvanlar üzerine etkisi nasıl?", "score": "C", "reasoning": "'Peki' ile devam ediyor, daha fazla ayrıntı istiyor."},
    {"message": "Gelecekte enerji ihtiyacını karşılamak için neler yapılabilir?", "score": "C", "reasoning": "Bir amaca ulaşmaya yönelik öneri istiyor, diyalogu ilerletiyor ama akıl yürütme yok."},
    {"message": "Çevre kirliliği ne gibi sorunlara yol açabilir", "score": "C", "reasoning": "Gelecek ile ilgili nedensel soru soruyor, diyalogu ilerletiyor ama akıl yürütme yok."},
    {"message": "yenilenemez enerjiler tükenseydi ne olurdu", "score": "C", "reasoning": "Gelecek ile ilgili varsayımsal soru soruyor, diyalogu ilerletiyor ama akıl yürütme yok."},
    {"message": "bunun için kullanılabilecek kağıt ve karton malzemeler çabuk ıslanır, cam da hızlı kırılabilir. yani bu seçenekleri kullanmak mantıksız", "score": "D", "reasoning": "Neden-sonuç ilişkisi ve akıl yürütme var."},
    {"message": "Geri dönüşüm yaparsak doğal kaynakları koruruz çünkü ham madde kullanımı azalır", "score": "D", "reasoning": "'Çünkü' ile açık neden-sonuç akıl yürütmesi."}
]

# --- FEATURE INDICATORS ---
REASONING_INDICATORS = [
    "çünkü", "bu nedenle", "dolayısıyla", "eğer", "ise", "o zaman", "olursa", "olmazsa",
    "sonuç olarak", "böylece", "nedeniyle", "sebebiyle", "bu sayede", "sayesinde", "yüzünden", "ötürü"
]

WEAK_REASONING = ["bu yüzden", "o yüzden"]
STRONG_ACTION = ["proje fikrim", "fikrim var", "planım", "önerim", "yapalım", "yapacağız"]

CONNECTION_INDICATORS = [
    "etkilenir", "etkiler", "etki", "ilişki", "bağlantı", "azalır", "azaltır", "artar", "artırır",
    "sebep olur", "neden olur", "sonucu", "sonucunda", "yerine", "karşılık", "ile birlikte", "beraber",
    "faydaları", "faydası", "yararları", "zararları"
]

QUESTION_WORDS = ["nedir", "ne demek", "nasıl", "neden", "niçin", "ne", "kim", "nerede", "ne zaman", "hangi"]
MINIMAL_INDICATORS = ["kısaca", "kısa", "özet", "özetçe"]
REFERENCE_INDICATORS = [
    "dediğin", "söylediğin", "bahsettiğin", "anlattığın", "peki", "ya", "başka", "ayrıca", "de ki", 
    "sen dediğim", "daha önce", "az önce", "yukarıda", "yaptığın üzerine"
]

CAUSAL_EFFECTUAL_INDICATORS = ["yapmazsak", "nelere yol açar", "gelecekte", "etkileri", "nedenleri", "sebepleri"]

ACTION_INDICATORS = [
    "yapabiliriz", "yapalım", "oluşturalım", "kuralım", "önerim", "proje", "plan", "adımlar", 
    "önce", "sonra", "fikrim", "düşünüyorum", "yapacağız", "yapmalıyız", "öneriyorum", "fikrim var"
]

MINIMAL_RESPONSES = ["evet", "hayır", "tamam", "ok", "olur", "anladım", "peki", "iyi"]
