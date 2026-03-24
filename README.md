# Hey Pakize — Wake Word Detection System

Derin Öğrenme dersi projesi. "Hey Pakize" uyandırma kelimesini mikrofondan gerçek zamanlı olarak tanıyan makine öğrenmesi sistemi.

**Ekip:** Hilal · Belinay · Hatice

---

## Proje Hakkında

Araç içi sesli asistanların otomatik açılmasını sağlayan bir uyandırma kelimesi tanıma sistemi. Mikrofon akışını dinler; "Hey Pakize" ifadesini duyduğunda terminal veya GUI üzerinden bildirim verir.

**Özellikler:**
- 39 boyutlu MFCC + Delta + Delta2 öznitelik çıkarımı
- 7 model karşılaştırması + GridSearchCV hiperparametre optimizasyonu
- 5-Fold StratifiedKFold çapraz doğrulama
- F1 bazlı eşik optimizasyonu
- Gerçek zamanlı terminal ve GUI modları
- Veri artırma (augmentation) ile genişletilmiş eğitim seti

---

## Kurulum

```bash
pip install librosa scikit-learn sounddevice soundfile numpy pandas matplotlib gtts
```

Python 3.8+ ve mikrofon gereklidir. TTS veri üretimi için internet bağlantısı gerekir.

---

## Klasör Yapısı

```
Hey-Pakize/
├── calistir.py                  # Tam pipeline çalıştırıcı (buradan başla)
│
├── src/
│   ├── tts_veri_uret.py         # gTTS ile sentetik ses üretimi
│   ├── isim_duzelt.py           # Metadata oluşturma + dev/test ayrımı
│   ├── oznitelik_cikarma.py     # 39 boyutlu MFCC öznitelik çıkarımı
│   ├── model_egitimi.py         # 7 model eğitimi + görseller
│   ├── analiz_veri.py           # Veri seti istatistikleri + görseller
│   └── wake_word_detector.py    # Gerçek zamanlı tespit (terminal + GUI)
│
├── data/
│   ├── raw/
│   │   ├── positive/            # El kaydı "Hey Pakize" örnekleri (.wav)
│   │   └── negative/            # El kaydı diğer sesler (.wav)
│   ├── tts/
│   │   ├── positive/            # TTS üretimi wake sesleri (otomatik)
│   │   └── negative/            # TTS üretimi nowake sesleri (otomatik)
│   ├── processed/
│   │   └── proje_veriseti.csv   # Çıkarılmış öznitelikler (otomatik)
│   └── metadata/
│       ├── all_files.csv        # Tüm ses dosyaları listesi (otomatik)
│       ├── dev_files.csv        # %80 geliştirme seti (otomatik)
│       └── test_files.csv       # %20 test seti (otomatik)
│
├── models/
│   └── improved/
│       ├── best_model.pkl       # En iyi eğitilmiş model (otomatik)
│       ├── scaler.pkl           # StandardScaler (otomatik)
│       └── model_metadata.json  # Model bilgileri ve eşik (otomatik)
│
└── results/
    └── visualizations/          # Tüm PNG grafikler (otomatik)
        ├── confusion_matrix.png
        ├── model_comparison.png
        ├── roc_curves.png
        ├── pr_curves.png
        ├── performans_tablosu.png
        ├── class_distribution.png
        ├── mfcc_distributions.png
        └── tsne.png
```

`(otomatik)` etiketli dosyalar pipeline tarafından üretilir, elle oluşturulmaz.

---

## Kullanım

### Tam Pipeline (Önerilen)

```bash
python calistir.py
```

Adımları sırasıyla çalıştırır ve her adımda ne yapıldığını gösterir:

1. **Temizlik** — Eski çıktılar silinir (onay istenir)
2. **TTS üretimi** (opsiyonel) — İnternet bağlantısı gerekir
3. **Metadata** — `data/raw/` ve `data/tts/` taranır, dev/test ayrımı yapılır
4. **Öznitelik çıkarma** — 39 boyutlu MFCC vektörleri CSV'ye yazılır
5. **Model eğitimi** — 7 model karşılaştırılır, en iyisi kaydedilir
6. **Veri analizi** (opsiyonel) — Ek görseller oluşturulur

### Tespiti Başlatma

```bash
# Terminal modu
python src/wake_word_detector.py

# GUI modu
python src/wake_word_detector.py --gui

# Özel eşik değeriyle
python src/wake_word_detector.py --gui -t 0.70
```

### Adımları Tek Tek Çalıştırma

```bash
python src/tts_veri_uret.py        # Sentetik ses üret
python src/isim_duzelt.py          # Metadata oluştur
python src/oznitelik_cikarma.py    # Öznitelik çıkar
python src/model_egitimi.py        # Modeli eğit
python src/analiz_veri.py          # Veri analizi
```

> **Not:** `oznitelik_cikarma.py` tek başına çalıştırıldığında mevcut CSV üzerine yazmaz.
> Yeniden oluşturmak için `calistir.py` kullanın.

---

## Ses Verisi

Ham kayıtlar `data/raw/` klasöründe saklanır:
- `data/raw/positive/` → "Hey Pakize" içeren kayıtlar
- `data/raw/negative/` → Diğer sesler (farklı kelimeler, cümleler, gürültü)

TTS ile üretilen sentetik sesler `data/tts/` altında tutulur ve ham kayıtlarla **karışmaz**.

Dosya isimlendirme kuralı:
- `tts_wake_XXX.wav` → sentetik pozitif
- `tts_nowake_XXX.wav` → sentetik negatif
- Diğer dosyalar: isimde `wake` veya `pakize` geçiyorsa pozitif, `nowake` veya `negatif` geçiyorsa negatif

### Veri Gizliligi

> **Not:** `data/raw/` klasöründeki ham ses kayıtlari bu repoda **paylasılmamaktadır.**
>
> Bu veri seti, Derin Öğrenme dersi kapsamında sınıf öğrencilerinin kendi seslerini kaydederek oluşturdukları kişisel verilerden oluşmaktadır. Kişisel ses kayıtlarının açık ortamda paylaşılması veri gizliliği açısından uygun olmadığından, ham ses dosyaları `.gitignore` aracılığıyla bu repoya dahil edilmemiştir.
>
> Projeyi çalıştırmak için kendi ses kayıtlarınızı `data/raw/positive/` ve `data/raw/negative/` klasörlerine yerleştirmeniz ya da `src/tts_veri_uret.py` ile sentetik veri üretmeniz gerekmektedir.

---

## Teknik Detaylar

### Öznitelikler (39 boyut)

| Grup | Boyut | Açıklama |
|------|-------|----------|
| MFCC ort | 13 | Spektral zarf — sesin genel rengi |
| Delta ort | 13 | MFCC değişim hızı — geçiş anları |
| Delta2 ort | 13 | MFCC değişim ivmesi — "Hey"/"Pakize" başlangıçları |

Ön işleme: pre-emphasis → sessizlik kesme → peak normalizasyon → 1.5 sn center padding

### Veri Artırma

| Kaynak | Varyasyon | Sebep |
|--------|-----------|-------|
| El kaydı pozitif | ×5 | Daha fazla çeşitlilik |
| TTS pozitif | ×1 | Sentetik ses zaten temiz |
| El kaydı negatif | ×2 | Orta düzey çeşitlilik |
| TTS negatif | ×1 | Sentetik ses zaten temiz |

Augmentation türleri: beyaz gürültü · hız değişimi · pitch kaydırma (yukarı/aşağı)

### Modeller

| Model | Notlar |
|-------|--------|
| Pos Benzerlik | Kosinüs benzerliği — sadece pozitif centroid öğrenir |
| SVM (RBF) | Doğrusal olmayan sınırlar |
| SVM (Linear) | Yüksek boyutlarda güçlü |
| Random Forest | Ensemble — overfitting'e dirençli |
| Gradient Boosting | Sıralı ensemble |
| Logistic Regression | Olasılıksal, yorumlanabilir |
| KNN | Örnek bazlı sınıflandırma |

Hepsi `class_weight="balanced"` ile eğitilir. En iyi model CV F1 skoruna göre seçilir.

### Gerçek Zamanlı Tespit

- Örnekleme hızı: 16 kHz · Pencere: 1.5 sn · Blok: 0.5 sn
- Son 3 tahminin ortalaması alınır (smoothing)
- RMS < 0.005 sessizlik → tahmin atlanır
- Eşik değeri: model metadata dosyasından okunur (varsayılan 0.65)

---

## Kullanılan Kütüphaneler

`librosa` · `scikit-learn` · `sounddevice` · `soundfile` · `numpy` · `pandas` · `matplotlib` · `tkinter` · `gtts` · `joblib`
