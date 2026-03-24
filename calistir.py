"""
calistir.py
-----------
Hey Pakize pipeline'ini sifirdan calistirir.

Her calistirmada once TUM eski ciktilar silinir:
  - data/processed/proje_veriseti.csv
  - models/improved/  (best_model.pkl, scaler.pkl, model_metadata.json)
  - results/visualizations/  (tum PNG'ler)

Sonra adim adim yeniden uretilir:
  1. TTS ses verisi   (src/tts_veri_uret.py)   [opsiyonel]
  2. Oznitelik cikarma (src/oznitelik_cikarma.py)
  3. Model egitimi    (src/model_egitimi.py)
  4. Veri analizi     (src/analiz_veri.py)      [opsiyonel]

Calistirma:
  python calistir.py
"""

import os, sys, subprocess
from pathlib import Path

BASE_DIR        = Path(__file__).resolve().parent
CSV_YOLU        = BASE_DIR / "data" / "processed" / "proje_veriseti.csv"
METADATA_KLASOR = BASE_DIR / "data" / "metadata"
MODEL_KLASOR    = BASE_DIR / "models" / "improved"
GORSEL_KLASOR   = BASE_DIR / "results" / "visualizations"
SRC             = BASE_DIR / "src"

MODEL_DOSYALARI    = ["best_model.pkl", "scaler.pkl", "model_metadata.json"]
METADATA_DOSYALARI = ["all_files.csv", "dev_files.csv", "test_files.csv"]


def sor(soru):
    try:
        yanit = input(soru + " [e/h]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return yanit in ("e", "evet", "y", "yes")


def silinecekleri_listele():
    """Mevcut eski ciktilari listeler, yoksa None doner."""
    liste = []
    if CSV_YOLU.exists():
        liste.append(str(CSV_YOLU))
    for d in MODEL_DOSYALARI:
        p = MODEL_KLASOR / d
        if p.exists():
            liste.append(str(p))
    for d in METADATA_DOSYALARI:
        p = METADATA_KLASOR / d
        if p.exists():
            liste.append(str(p))
    for png in GORSEL_KLASOR.glob("*.png"):
        liste.append(str(png))
    return liste


def eski_ciktilari_sil():
    silinen = 0

    if CSV_YOLU.exists():
        CSV_YOLU.unlink()
        print("    ✓ proje_veriseti.csv silindi.")
        silinen += 1

    for d in MODEL_DOSYALARI:
        p = MODEL_KLASOR / d
        if p.exists():
            p.unlink()
            print(f"    ✓ {d} silindi.")
            silinen += 1

    for d in METADATA_DOSYALARI:
        p = METADATA_KLASOR / d
        if p.exists():
            p.unlink()
            print(f"    ✓ metadata/{d} silindi.")
            silinen += 1

    png_sayisi = 0
    for png in GORSEL_KLASOR.glob("*.png"):
        png.unlink()
        png_sayisi += 1
    if png_sayisi:
        print(f"    ✓ {png_sayisi} adet PNG gorseli silindi.")
        silinen += png_sayisi

    if silinen == 0:
        print("    (Silinecek eski cikti bulunamadi — temiz baslangic.)")
    return silinen


def adim_calistir(script_adi, aciklama):
    print(f"\n{'─'*60}")
    print(f"  {aciklama}")
    print(f"{'─'*60}")
    sonuc = subprocess.run([sys.executable, str(SRC / script_adi)])
    if sonuc.returncode != 0:
        print(f"\n[HATA] '{script_adi}' basarisiz (kod {sonuc.returncode}).")
        print("Pipeline durduruldu. Hataji giderip tekrar calistirin.")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  HEY PAKIZE — Tam Pipeline Calistirici")
    print("=" * 60)

    # ── Eski ciktilari temizle ─────────────────────────────────
    print("\n[Temizlik] Eski ciktilar kontrol ediliyor...")
    silinecekler = silinecekleri_listele()

    if silinecekler:
        print(f"\n  Asagidaki {len(silinecekler)} dosya silinecek:")
        for s in silinecekler:
            print(f"    - {s}")
        print()
        if not sor("  Devam edilsin mi? (ONAY GEREKMEZ — bu her zaman yapilir)"):
            print("\nIptal edildi.")
            sys.exit(0)
    else:
        print("  Temiz baslangic — silinecek eski cikti yok.")

    print("\n  Siliniyor...")
    eski_ciktilari_sil()
    print("  ✓ Temizlik tamamlandi.")

    # ── Adim 1: TTS veri uretimi (opsiyonel) ──────────────────
    print("\n[Adim 1] TTS Ses Verisi Uretimi")
    print("  Ham ses dosyalari (data/raw/) SILINMEZ — yalnizca yeni TTS eklenir.")
    if sor("  Yeni TTS sesi uretilsin mi?"):
        adim_calistir("tts_veri_uret.py", "TTS ses verisi uretiliyor...")
    else:
        print("  Atlandı — mevcut ham ses dosyalari kullanilacak.")

    # ── Adim 2: Metadata olusturma ─────────────────────────────
    # TTS'den sonra calisir: yeni dosyalar da metadata'ya girer.
    # Ciktilar: data/metadata/all_files.csv, dev_files.csv, test_files.csv
    adim_calistir("isim_duzelt.py",
                  "Metadata olusturuluyor... (tum ham sesler listeleniyor, dev/test ayrimi)")

    # ── Adim 3: Oznitelik cikarma ──────────────────────────────
    adim_calistir("oznitelik_cikarma.py",
                  "Oznitelikler cikartiliyor... (tum ham sesler isleniyor)")

    # ── Adim 4: Model egitimi ──────────────────────────────────
    adim_calistir("model_egitimi.py",
                  "Model egitiliyor... (birkaç dakika surebilir)")

    # ── Adim 5: Veri analizi (opsiyonel) ──────────────────────
    print(f"\n{'─'*60}")
    print("  [Adim 5] Veri Analizi Gorselleri")
    print(f"{'─'*60}")
    if sor("  Ek veri analizi gorselleri olusturulsun mu?"):
        adim_calistir("analiz_veri.py", "Veri analizi yapiliyor...")
    else:
        print("  Atlandı.")

    # ── Bitis ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Pipeline basariyla tamamlandi!")
    print()
    print("  Olusturulan ciktilar:")
    print(f"    CSV    : {CSV_YOLU}")
    print(f"    Model  : {MODEL_KLASOR}/best_model.pkl")
    print(f"    Gorsel : {GORSEL_KLASOR}/")
    print()
    print("  Detektoru baslatmak icin:")
    print("    python src/wake_word_detector.py --gui")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nKullanici tarafindan iptal edildi.")
        sys.exit(0)
