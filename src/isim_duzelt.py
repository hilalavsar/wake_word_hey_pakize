# -*- coding: utf-8 -*-
"""
isim_duzelt.py
--------------
Pozitif ve negatif ses dosyalarını bulur, metadata oluşturur ve
veriyi sızıntısız biçimde ayırır.

Çıktılar:
  data/metadata/all_files.csv
  data/metadata/dev_files.csv
  data/metadata/test_files.csv

Taranan klasörler:
  data/raw/positive/   data/raw/negative/    ← el kaydı (ham)
  data/tts/positive/   data/tts/negative/    ← sentetik (TTS)

Not:
- Pozitif: hedef kelime / wake word kayıtları
- Negatif: hedef kelime dışındaki kayıtlar
"""

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================
# AYARLAR
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

POSITIVE_DIRS = [
    BASE_DIR / "data" / "raw" / "positive",
    BASE_DIR / "data" / "tts" / "positive",
]
NEGATIVE_DIRS = [
    BASE_DIR / "data" / "raw" / "negative",
    BASE_DIR / "data" / "tts" / "negative",
]

METADATA_DIR = BASE_DIR / "data" / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20   # %20 final test
VALID_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def list_audio_files(folder: Path) -> list[Path]:
    """Klasördeki tüm ses dosyalarını recursive olarak bulur."""
    if not folder.exists():
        return []
    files = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            files.append(path)
    return sorted(files)


def build_metadata() -> pd.DataFrame:
    """Pozitif ve negatif kayıtları etiketleyip tek tabloda toplar."""
    rows = []

    for folder in POSITIVE_DIRS:
        for fp in list_audio_files(folder):
            rows.append({
                "filepath": str(fp),
                "label": 1,
                "class_name": "positive",
                "filename": fp.name
            })

    for folder in NEGATIVE_DIRS:
        for fp in list_audio_files(folder):
            rows.append({
                "filepath": str(fp),
                "label": 0,
                "class_name": "negative",
                "filename": fp.name
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "Hiç ses dosyası bulunamadı.\n"
            f"Taranan pozitif klasörler: {POSITIVE_DIRS}\n"
            f"Taranan negatif klasörler: {NEGATIVE_DIRS}"
        )

    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Veriyi:
      - %80 development
      - %20 final test
    şeklinde ayırır.

    Development set daha sonra model eğitimi sırasında
    StratifiedKFold ile iç doğrulama için kullanılacak.
    """
    dev_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return dev_df, test_df


def print_summary(df: pd.DataFrame, name: str) -> None:
    total = len(df)
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Toplam:   {total}")
    print(f"Pozitif:  {pos}")
    print(f"Negatif:  {neg}")
    if total > 0:
        print(f"Pozitif oranı: %{100 * pos / total:.2f}")
        print(f"Negatif oranı: %{100 * neg / total:.2f}")


def main() -> None:
    print("Metadata oluşturuluyor...")
    df = build_metadata()

    all_csv = METADATA_DIR / "all_files.csv"
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")

    dev_df, test_df = stratified_split(df)

    dev_csv = METADATA_DIR / "dev_files.csv"
    test_csv = METADATA_DIR / "test_files.csv"

    dev_df.to_csv(dev_csv, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_csv, index=False, encoding="utf-8-sig")

    print_summary(df, "TÜM VERİ")
    print_summary(dev_df, "DEVELOPMENT SET (%80)")
    print_summary(test_df, "FINAL TEST SET (%20)")

    print("\nDosyalar kaydedildi:")
    print(f"  {all_csv}")
    print(f"  {dev_csv}")
    print(f"  {test_csv}")


if __name__ == "__main__":
    main()