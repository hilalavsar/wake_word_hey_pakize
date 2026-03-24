"""
tts_veri_uret.py
----------------
gTTS ile Türkçe ses kayıtları üretir.

Wake (pozitif):
  - "Hey Pakize" farklı hız/pitch varyasyonlarıyla

Nowake (negatif):
  - "Hey" tek başına
  - "Pakize" tek başına
  - "Hey Fatih", "Hey Ahmet" vb. benzer kelimeler
  - Farklı cümleler

Çıktı klasörü:
  data/tts/positive/   ← Ham kayıtlara karışmaz (data/raw/ ayrı kalır)
  data/tts/negative/

Kurulum:
  pip install gtts librosa soundfile numpy

Çalıştırma:
  python tts_veri_uret.py
"""

import os
import io
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import warnings
warnings.filterwarnings("ignore")

try:
    from gtts import gTTS
except ImportError:
    print("gTTS yüklü değil. Şunu çalıştır: pip install gtts")
    exit(1)

# ── Ayarlar ───────────────────────────────────────────────
SR          = 16000
SURE        = 1.5          # oznitelik_cikarma ile tutarlı
MAX_SAMPLES = int(SR * SURE)

BASE_DIR         = Path(__file__).resolve().parent.parent
POZITIF_KLASOR   = BASE_DIR / "data" / "tts" / "positive"
NEGATIF_KLASOR   = BASE_DIR / "data" / "tts" / "negative"

POZITIF_KLASOR.mkdir(parents=True, exist_ok=True)
NEGATIF_KLASOR.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────────────────

# ── Wake kelimeler ────────────────────────────────────────
WAKE_METINLER = [
    "Hey Pakize",
    "Hey Pakize",
    "Hey Pakize",
    "Hey Pakize",
    "Hey Pakize",
    "Hey Pakize",
    "Hey, Pakize",
    "Hey, Pakize",
    "heyy Pakize",
    "Heyy Pakize",
    "Hey Pakize lütfen",
    "Hay Pakize",
]

# ── Nowake kelimeler ──────────────────────────────────────
NOWAKE_METINLER = [
    # Sadece "Hey"
    "Hey",
    "Hey",
    "hey",
    # Sadece "Pakize"
    "Pakize",
    "Pakize",
    # Benzer wake kelimeler (Hard Negatives)
    "Hey Fatih",
    "Hey Ahmet",
    "Hey Ayşe",
    "Hey Mehmet",
    "Hey Zeynep",
    "Hey Pazar",
    "Hey Paket",
    "Hey Patron",
    "Hey siri",
    "Hey google",
    # Hey + 'A' ile başlayan kelimeler (Pakize'ye fonetik benzerlik)
    "Hey Araba",
    "Hey Adem",
    "Hey Ali",
    "Hey Ayhan",
    "Hey Asistan",
    # Hey + 'Pa' ile başlayan kelimeler (çok kritik hard negative)
    "Hey Parti",
    "Hey Pasta",
    "Hey Para",
    # Farklı cümleler
    "Merhaba nasılsın",
    "Tamam anladım",
    "Evet hayır",
    "Bir iki üç",
    "Bugün hava güzel",
    "Ne zaman gidiyorsun",
    "Lütfen dur",
    "Teşekkür ederim",
]


def tts_ses_uret(metin, lang="tr"):
    """gTTS ile metin → ses (numpy array)."""
    tts = gTTS(text=metin, lang=lang, slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    y, _ = librosa.load(buf, sr=SR, mono=True)
    return y


def sabitle_ve_kaydet(y, dosya_yolu):
    """Sesi 2 sn'ye sabitle, ortaya hizala ve kaydet."""
    # Peak normalization
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Center padding
    if len(y) >= MAX_SAMPLES:
        start = (len(y) - MAX_SAMPLES) // 2
        y = y[start:start + MAX_SAMPLES]
    else:
        pad = MAX_SAMPLES - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))

    sf.write(dosya_yolu, y, SR)


def augment_tts(y, hiz=1.0, pitch=0, gurultu=0.0):
    """TTS sesine hız/pitch/gürültü varyasyonu uygula."""
    if hiz != 1.0:
        y = librosa.effects.time_stretch(y, rate=hiz)
    if pitch != 0:
        y = librosa.effects.pitch_shift(y, sr=SR, n_steps=pitch)
    if gurultu > 0:
        y = y + gurultu * np.random.randn(len(y)).astype(np.float32)
    return y


def _sonraki_sayac(klasor, prefix):
    """Klasördeki mevcut dosyaların numarasından sonrasını döndür."""
    mevcutlar = list(klasor.glob(f"{prefix}_*.wav"))
    if not mevcutlar:
        return 0
    numaralar = []
    for f in mevcutlar:
        try:
            numaralar.append(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    return max(numaralar) + 1 if numaralar else 0


def uret_wake():
    """Wake (Hey Pakize) seslerini üret."""
    print("\n── Wake sesler üretiliyor ──")
    sayac = _sonraki_sayac(POZITIF_KLASOR, "tts_wake")

    # Hız ve pitch varyasyonları
    varyasyonlar = [
        (1.0,  0,   0.0),    # normal
        (0.85, 0,   0.0),    # yavaş
        (1.15, 0,   0.0),    # hızlı
        (1.0,  1,   0.0),    # yüksek ton
        (1.0, -1,   0.0),    # alçak ton
        (1.0,  0,   0.02),   # gürültülü
        (0.9,  1,   0.01),   # yavaş + yüksek
        (1.1, -1,   0.01),   # hızlı + alçak
    ]

    for i, metin in enumerate(WAKE_METINLER):
        try:
            y_base = tts_ses_uret(metin)

            for (hiz, pitch, gurultu) in varyasyonlar:
                y_aug = augment_tts(y_base, hiz=hiz, pitch=pitch, gurultu=gurultu)
                dosya = POZITIF_KLASOR / f"tts_wake_{sayac:03d}.wav"
                sabitle_ve_kaydet(y_aug, str(dosya))
                sayac += 1

            print(f"  ✓ '{metin}' → {len(varyasyonlar)} varyasyon")
        except Exception as e:
            print(f"  ✗ Hata ({metin}): {e}")

    print(f"  Toplam {sayac} wake ses üretildi.")
    return sayac


def uret_nowake():
    """Nowake seslerini üret."""
    print("\n── Nowake sesler üretiliyor ──")
    sayac = _sonraki_sayac(NEGATIF_KLASOR, "tts_nowake")

    varyasyonlar = [
        (1.0,  0,   0.0),
        (0.9,  0,   0.0),
        (1.1,  0,   0.02),
    ]

    for metin in NOWAKE_METINLER:
        try:
            y_base = tts_ses_uret(metin)

            for (hiz, pitch, gurultu) in varyasyonlar:
                y_aug = augment_tts(y_base, hiz=hiz, pitch=pitch, gurultu=gurultu)
                dosya = NEGATIF_KLASOR / f"tts_nowake_{sayac:03d}.wav"
                sabitle_ve_kaydet(y_aug, str(dosya))
                sayac += 1

            print(f"  ✓ '{metin}' → {len(varyasyonlar)} varyasyon")
        except Exception as e:
            print(f"  ✗ Hata ({metin}): {e}")

    print(f"  Toplam {sayac} nowake ses üretildi.")
    return sayac


def main():
    print("=" * 55)
    print("  TTS Veri Üretimi — Hey Pakize Projesi")
    print("=" * 55)
    print(f"Pozitif cikti : {POZITIF_KLASOR}")
    print(f"Negatif cikti : {NEGATIF_KLASOR}")
    print("Not: İnternet bağlantısı gereklidir (gTTS).\n")

    wake_sayisi   = uret_wake()
    nowake_sayisi = uret_nowake()

    print(f"\n{'─'*55}")
    print(f"  Üretilen wake   : {wake_sayisi}")
    print(f"  Üretilen nowake : {nowake_sayisi}")
    print(f"  TOPLAM          : {wake_sayisi + nowake_sayisi}")
    print("\nSimdi sirayla calistir:")
    print("  python src/oznitelik_cikarma.py")
    print("  python src/model_egitimi.py")
    print("  python src/wake_word_detector.py --gui")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()