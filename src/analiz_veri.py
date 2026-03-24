"""
analiz_veri.py
--------------
Veri seti istatistiklerini ve görselleştirmeleri oluşturur:
  - Sınıf dağılımı
  - MFCC dağılım boxplot
  - t-SNE görselleştirme (pozitif / negatif ayrımı)

Çalıştırmadan önce oznitelik_cikarma.py çalıştırılmış olmalı.
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR      = Path(__file__).resolve().parent.parent
CSV_YOLU      = str(BASE_DIR / "data" / "processed" / "proje_veriseti.csv")
GORSEL_KLASOR = str(BASE_DIR / "results" / "visualizations")
os.makedirs(GORSEL_KLASOR, exist_ok=True)


def temel_istatistikler(df):
    toplam = len(df)
    pozitif = (df["etiket"] == 1).sum()
    negatif = (df["etiket"] == 0).sum()
    oran = negatif / pozitif if pozitif > 0 else float("inf")

    print("=" * 55)
    print("  VERİ SETİ TEMEL İSTATİSTİKLERİ")
    print("=" * 55)
    print(f"\n  Toplam Örnek           : {toplam}")
    print(f"  ├─ Pozitif (Hey Pakize): {pozitif}  (%{100*pozitif/toplam:.1f})")
    print(f"  └─ Negatif (Diğer)     : {negatif}  (%{100*negatif/toplam:.1f})")
    print(f"\n  Denge Oranı            : 1:{oran:.2f} (Pozitif:Negatif)")
    oz_sutunlari = [c for c in df.columns if c not in ("dosya", "etiket")]
    n_mfcc  = sum(1 for c in oz_sutunlari if c.startswith("mfcc_"))
    n_delta = sum(1 for c in oz_sutunlari if c.startswith("delta_"))
    n_d2    = sum(1 for c in oz_sutunlari if c.startswith("delta2_"))
    print(f"  Öznitelik Sayısı       : {len(oz_sutunlari)}"
          f"  ({n_mfcc} MFCC + {n_delta} Delta + {n_d2} Delta2)")
    print()


def sinif_dagilim_gorseli(df):
    etiketler = ["Negatif (nowake)", "Pozitif (wake)"]
    sayilar   = [(df["etiket"] == 0).sum(), (df["etiket"] == 1).sum()]
    renkler   = ["#4A90D9", "#E8593C"]

    fig, ax = plt.subplots(figsize=(5, 4))
    cubuklar = ax.bar(etiketler, sayilar, color=renkler, width=0.5, edgecolor="white")
    for c, s in zip(cubuklar, sayilar):
        ax.text(c.get_x() + c.get_width()/2, c.get_height() + 3,
                str(s), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Sınıf Dağılımı")
    ax.set_ylabel("Örnek Sayısı")
    ax.set_ylim(0, max(sayilar) * 1.2)
    plt.tight_layout()
    yol = os.path.join(GORSEL_KLASOR, "class_distribution.png")
    plt.savefig(yol, dpi=120)
    plt.close()
    print(f"  ✓ Sınıf dağılımı → {yol}")


def mfcc_boxplot(df):
    oz_sutunlari = [c for c in df.columns if c.startswith("mfcc_")]
    if not oz_sutunlari:
        print("  ⚠ MFCC sütunları bulunamadı.")
        return

    df_pos = df[df["etiket"] == 1][oz_sutunlari]
    df_neg = df[df["etiket"] == 0][oz_sutunlari]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_pos.boxplot(ax=axes[0], vert=True)
    axes[0].set_title("MFCC Dağılımı — Pozitif (Hey Pakize)")
    axes[0].set_xticklabels([f"M{i+1}" for i in range(len(oz_sutunlari))], rotation=45)

    df_neg.boxplot(ax=axes[1], vert=True)
    axes[1].set_title("MFCC Dağılımı — Negatif")
    axes[1].set_xticklabels([f"M{i+1}" for i in range(len(oz_sutunlari))], rotation=45)

    plt.tight_layout()
    yol = os.path.join(GORSEL_KLASOR, "mfcc_distributions.png")
    plt.savefig(yol, dpi=120)
    plt.close()
    print(f"  ✓ MFCC dağılımı  → {yol}")


def tsne_gorsel(df):
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.manifold      import TSNE
    except ImportError:
        print("  ⚠ scikit-learn yok, t-SNE atlandı.")
        return

    oz_sutunlari = [c for c in df.columns if c not in ("dosya", "etiket")]
    X = df[oz_sutunlari].values
    y = df["etiket"].values

    # Büyük veri setlerinde hızlı tut
    if len(X) > 500:
        idx = np.random.choice(len(X), 500, replace=False)
        X, y = X[idx], y[idx]

    X_scaled = StandardScaler().fit_transform(X)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 6))
    for etiket, renk, ad in [(0, "#4A90D9", "Negatif"), (1, "#E8593C", "Pozitif")]:
        mask = y == etiket
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=renk, label=ad, alpha=0.6, s=25, edgecolors="none")
    ax.set_title("t-SNE — Öznitelik Uzayı (2D)")
    ax.legend()
    plt.tight_layout()
    yol = os.path.join(GORSEL_KLASOR, "tsne.png")
    plt.savefig(yol, dpi=120)
    plt.close()
    print(f"  ✓ t-SNE            → {yol}")


def main():
    if not os.path.exists(CSV_YOLU):
        print(f"HATA: '{CSV_YOLU}' bulunamadı.")
        print("Önce oznitelik_cikarma.py çalıştır!")
        return

    df = pd.read_csv(CSV_YOLU)
    temel_istatistikler(df)

    print("Görseller oluşturuluyor...")
    sinif_dagilim_gorseli(df)
    mfcc_boxplot(df)
    tsne_gorsel(df)
    print("\nTamamlandı!")


if __name__ == "__main__":
    main()
