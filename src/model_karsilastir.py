"""
model_karsilastir.py
--------------------
İki farklı model dosyasını AYNI test seti üzerinde karşılaştırır.

Kullanım:
  python src/model_karsilastir.py --a models/improved --b models/eski_model

Klasör yapısı (her model klasörü için):
  klasor/best_model.pkl
  klasor/scaler.pkl
  klasor/model_metadata.json   (opsiyonel, isim için)
"""

import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (f1_score, accuracy_score, precision_score,
                              recall_score, roc_curve, auc,
                              confusion_matrix, classification_report)

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_YOLU = BASE_DIR / "data" / "processed" / "proje_veriseti.csv"
CIKTI    = BASE_DIR / "results" / "visualizations" / "model_karsilastirma.png"


def model_yukle(klasor):
    klasor = Path(klasor)
    model  = joblib.load(klasor / "best_model.pkl")
    scaler = joblib.load(klasor / "scaler.pkl")
    isim   = klasor.name
    meta_yolu = klasor / "model_metadata.json"
    if meta_yolu.exists():
        with open(meta_yolu, encoding="utf-8") as f:
            meta = json.load(f)
        isim = f"{meta.get('model_adi','?')} ({klasor.name})"
    return model, scaler, isim


def degerlendir(model, scaler, X_test, y_test, isim):
    proba = model.predict_proba(scaler.transform(X_test))[:, 1]

    # F1 bazlı en iyi eşik
    en_iyi_f1, en_iyi_esik = 0, 0.5
    for t in np.arange(0.25, 0.96, 0.05):
        yp = (proba >= t).astype(int)
        f  = f1_score(y_test, yp, zero_division=0)
        if f > en_iyi_f1:
            en_iyi_f1, en_iyi_esik = f, t

    yp = (proba >= en_iyi_esik).astype(int)
    return {
        "isim":      isim,
        "accuracy":  round(accuracy_score(y_test, yp), 4),
        "precision": round(precision_score(y_test, yp, zero_division=0), 4),
        "recall":    round(recall_score(y_test, yp, zero_division=0), 4),
        "f1":        round(f1_score(y_test, yp, zero_division=0), 4),
        "esik":      round(en_iyi_esik, 2),
        "proba":     proba,
        "y_pred":    yp,
    }


def goster(r_a, r_b, y_test):
    print("\n" + "=" * 60)
    print(f"  MODEL A : {r_a['isim']}")
    print(f"  MODEL B : {r_b['isim']}")
    print("=" * 60)
    print(f"{'Metrik':<14} {'Model A':>10} {'Model B':>10} {'Fark':>10}")
    print("-" * 46)
    for m in ("accuracy", "precision", "recall", "f1"):
        fark = r_b[m] - r_a[m]
        ok   = "▲" if fark > 0 else ("▼" if fark < 0 else "=")
        print(f"{m:<14} {r_a[m]:>10.4f} {r_b[m]:>10.4f} {ok} {abs(fark):.4f}")
    print(f"{'esik':<14} {r_a['esik']:>10.2f} {r_b['esik']:>10.2f}")
    print()

    kazanan = r_a if r_a["f1"] >= r_b["f1"] else r_b
    print(f"  Kazanan (F1): {kazanan['isim']}  →  F1={kazanan['f1']}")

    print(f"\n--- {r_a['isim']} ---")
    print(classification_report(y_test, r_a["y_pred"],
                                 target_names=["Negatif", "Pozitif"]))
    print(f"--- {r_b['isim']} ---")
    print(classification_report(y_test, r_b["y_pred"],
                                 target_names=["Negatif", "Pozitif"]))


def gorsel_kaydet(r_a, r_b, y_test):
    renkler = {"A": "#4A90D9", "B": "#E8593C"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── ROC ──────────────────────────────────────────────
    ax = axes[0]
    for r, harf, renk in [(r_a, "A", renkler["A"]), (r_b, "B", renkler["B"])]:
        fpr, tpr, _ = roc_curve(y_test, r["proba"])
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=renk, lw=2,
                label=f"{harf}: {r['isim'][:20]}  (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Eğrileri"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=8)

    # ── Metrik Karşılaştırma ──────────────────────────────
    ax = axes[1]
    metrikler = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrikler))
    ax.bar(x - 0.2, [r_a[m] for m in metrikler], 0.4,
           label=f"A: {r_a['isim'][:20]}", color=renkler["A"])
    ax.bar(x + 0.2, [r_b[m] for m in metrikler], 0.4,
           label=f"B: {r_b['isim'][:20]}", color=renkler["B"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.2)
    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8)
    ax.set_title("Metrik Karşılaştırması")
    ax.legend(fontsize=8)

    # ── Confusion Matrix (Fark) ───────────────────────────
    ax = axes[2]
    cm_a = confusion_matrix(y_test, r_a["y_pred"])
    cm_b = confusion_matrix(y_test, r_b["y_pred"])
    fark = cm_b.astype(int) - cm_a.astype(int)
    im = ax.imshow(fark, cmap="RdYlGn", vmin=-10, vmax=10)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negatif", "Pozitif"])
    ax.set_yticklabels(["Negatif", "Pozitif"])
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix Farkı\n(B − A, yeşil=B daha iyi)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{fark[i,j]:+d}", ha="center", va="center",
                    fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax)

    plt.suptitle("Model A vs Model B — Aynı Test Seti", fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(CIKTI.parent, exist_ok=True)
    plt.savefig(CIKTI, dpi=130)
    plt.close()
    print(f"\n  Görsel kaydedildi → {CIKTI}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Model A klasörü (örn: models/improved)")
    ap.add_argument("--b", required=True, help="Model B klasörü (örn: models/eski_model)")
    ap.add_argument("--test-orani", type=float, default=0.20,
                    help="Test seti oranı (varsayılan 0.20)")
    args = ap.parse_args()

    if not CSV_YOLU.exists():
        print(f"HATA: '{CSV_YOLU}' bulunamadı. Önce oznitelik_cikarma.py çalıştır!")
        return

    # Veri
    df = pd.read_csv(CSV_YOLU)
    X  = df.drop(columns=["dosya", "etiket"]).values.astype(np.float32)
    y  = df["etiket"].values
    print(f"Veri: {len(y)} örnek  (Pozitif: {(y==1).sum()}  Negatif: {(y==0).sum()})")

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_orani, random_state=42, stratify=y)
    print(f"Test seti: {len(y_test)} örnek\n")

    # Modeller
    model_a, scaler_a, isim_a = model_yukle(args.a)
    model_b, scaler_b, isim_b = model_yukle(args.b)

    r_a = degerlendir(model_a, scaler_a, X_test, y_test, isim_a)
    r_b = degerlendir(model_b, scaler_b, X_test, y_test, isim_b)

    goster(r_a, r_b, y_test)
    gorsel_kaydet(r_a, r_b, y_test)


if __name__ == "__main__":
    main()
