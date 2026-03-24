"""
oznitelik_cikarma.py
--------------------
39 boyutlu öznitelik çıkarımı.

Öznitelikler:
  13 MFCC ort + 13 Delta ort + 13 Delta2 ort = 39 boyut

Ön işleme:
  ✓ Pre-emphasis (0.97)     → H, P, K ünsüzleri güçlenir
  ✓ Silence trim (30dB)     → baştaki/sondaki boşluklar kesilir
  ✓ Peak normalization      → ses seviyesi farkı giderilir
  ✓ Center padding (1.5 sn) → hızlı/yavaş/aralıklı sesler ortaya hizalanır

Augmentation:
  Gerçek pozitif → 5 varyasyon
  TTS pozitif    → 1 varyasyon  (sentetik ses zaten temiz, fazla aug overfit yapar)
  Gerçek negatif → 2 varyasyon
  TTS negatif    → 1 varyasyon

Kaynak dizinler:
  data/raw/positive/   data/raw/negative/    ← el kaydı (ham)
  data/tts/positive/   data/tts/negative/    ← sentetik (TTS)
"""

import os, sys, glob, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent

KLASORLER = [
    str(BASE_DIR / "data" / "raw"  / "negative"),
    str(BASE_DIR / "data" / "raw"  / "positive"),
    str(BASE_DIR / "data" / "tts"  / "negative"),
    str(BASE_DIR / "data" / "tts"  / "positive"),
]

CIKTI_CSV = str(BASE_DIR / "data" / "processed" / "proje_veriseti.csv")

SR          = 16000
SURE        = 1.5
N_MFCC      = 13
HOP_LENGTH  = 512
N_FFT       = 2048
MAX_SAMPLES = int(SR * SURE)
POS_AUG     = 5   # gerçek insan kaydı
NEG_AUG     = 2   # gerçek insan kaydı
TTS_AUG     = 1   # TTS ses (sentetik)
# Öznitelik boyutu: 13 MFCC + 13 Delta + 13 Delta2 = 39


def pre_emphasis(y, coef=0.97):
    return np.append(y[0], y[1:] - coef * y[:-1])


def yukle_ve_isle(dosya_yolu):
    y, _ = librosa.load(dosya_yolu, sr=SR, mono=True)
    y = pre_emphasis(y)
    y_trim, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trim) < SR * 0.3:
        y_trim = y
    peak = np.max(np.abs(y_trim))
    if peak > 0:
        y_trim = y_trim / peak
    if len(y_trim) >= MAX_SAMPLES:
        start = (len(y_trim) - MAX_SAMPLES) // 2
        y_out = y_trim[start:start + MAX_SAMPLES]
    else:
        pad = MAX_SAMPLES - len(y_trim)
        y_out = np.pad(y_trim, (pad // 2, pad - pad // 2))
    return y_out.astype(np.float32)


def oznitelik_cikart(y):
    mfcc   = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC,
                                   hop_length=HOP_LENGTH, n_fft=N_FFT)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc,   axis=1),   # 13 — spektral zarf
        np.mean(delta,  axis=1),   # 13 — geçiş hızı
        np.mean(delta2, axis=1),   # 13 — geçiş ivmesi (Hey/Pakize başlangıçları)
    ]).astype(np.float32)          # = 39


def augment_ses(y, n):
    varyasyonlar = []
    nf = np.random.uniform(0.02, 0.05)
    varyasyonlar.append(("noise", (y + nf * np.random.randn(len(y))).astype(np.float32)))
    if n >= 2:
        try:
            rate = np.random.uniform(0.85, 1.15)
            ys = librosa.effects.time_stretch(y, rate=rate)
            pad = MAX_SAMPLES - len(ys)
            ys = np.pad(ys, (max(0, pad//2), max(0, pad-pad//2)))[:MAX_SAMPLES]
            varyasyonlar.append(("speed", ys.astype(np.float32)))
        except Exception:
            pass
    if n >= 3:
        try:
            varyasyonlar.append(("pitch_up",
                librosa.effects.pitch_shift(y, sr=SR, n_steps=1.5).astype(np.float32)))
        except Exception:
            pass
    if n >= 4:
        try:
            varyasyonlar.append(("pitch_dn",
                librosa.effects.pitch_shift(y, sr=SR, n_steps=-1.5).astype(np.float32)))
        except Exception:
            pass
    return varyasyonlar[:n]


def isle(zorla=False):
    if not zorla and os.path.exists(CIKTI_CSV):
        print(f"\n⚠  '{CIKTI_CSV}' zaten mevcut!")
        print("   Üstüne yazmak için calistir.py kullanın.")
        print("   (calistir.py eski CSV'yi silip onayınızla yeniden oluşturur)")
        sys.exit(0)
    np.random.seed(42)
    os.makedirs(os.path.dirname(CIKTI_CSV), exist_ok=True)

    dosyalar = []
    for k in KLASORLER:
        dosyalar.extend(glob.glob(os.path.join(k, "*.wav")))
    dosyalar = sorted(dosyalar)

    if not dosyalar:
        print(f"HATA: WAV bulunamadı: {KLASORLER}")
        return

    print(f"{len(dosyalar)} ses dosyası işlenecek...\n")

    sutunlar = (
        [f"mfcc_{i+1}"   for i in range(N_MFCC)] +
        [f"delta_{i+1}"  for i in range(N_MFCC)] +
        [f"delta2_{i+1}" for i in range(N_MFCC)]
    )

    satirlar, basarili, atlanan = [], 0, 0

    for dosya in dosyalar:
        ad = os.path.basename(dosya).lower()
        is_tts = ad.startswith("tts_")

        if "nowake" in ad or "negatif" in ad or "tts_nowake" in ad:
            etiket = 0
        elif "wake" in ad or "pakize" in ad or "pozitif" in ad or "tts_wake" in ad:
            etiket = 1
        else:
            print(f"  Atlandı: {ad}")
            atlanan += 1
            continue

        try:
            y  = yukle_ve_isle(dosya)
            of = oznitelik_cikart(y)
            satir = dict(zip(sutunlar, of))
            satir["dosya"]  = ad
            satir["etiket"] = etiket
            satirlar.append(satir)
            basarili += 1

            if is_tts:
                n_aug = TTS_AUG
            elif etiket == 1:
                n_aug = POS_AUG
            else:
                n_aug = NEG_AUG

            for aug_ad, y_aug in augment_ses(y, n_aug):
                of_aug = oznitelik_cikart(y_aug)
                aug_satir = dict(zip(sutunlar, of_aug))
                aug_satir["dosya"]  = f"aug_{aug_ad}_{ad}"
                aug_satir["etiket"] = etiket
                satirlar.append(aug_satir)

        except Exception as e:
            print(f"  Hata: {ad} → {e}")
            atlanan += 1

    df = pd.DataFrame(satirlar)
    oz = [c for c in df.columns if c not in ("dosya", "etiket")]
    df = df[oz + ["dosya", "etiket"]]
    df.to_csv(CIKTI_CSV, index=False)

    pos = (df["etiket"] == 1).sum()
    neg = (df["etiket"] == 0).sum()
    print(f"{'─'*55}")
    print(f"  İşlenen          : {basarili}  |  Atlanan: {atlanan}")
    print(f"  Pozitif (wake)   : {pos}")
    print(f"  Negatif (nowake) : {neg}")
    if pos > 0:
        print(f"  Oran             : 1:{neg/pos:.2f}")
    print(f"  Toplam           : {len(df)}")
    print(f"  Öznitelik boyutu : {len(oz)}")
    print(f"  CSV              : {CIKTI_CSV}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    isle()
