"""
model_egitimi.py
----------------
En iyi model eğitimi — tek aşamalı, güçlü parametreler.

Neden tek aşamaya döndük:
  TTS verisiyle artık "Hey", "Pakize", "Hey Fatih" nowake olarak
  veri setinde var. Model bunları artık öğrenebilir.
  İki aşamalı sistem aynı veriyle eğitildiğinde ek fayda sağlamıyor.

İyileştirmeler:
  ✓ 6 model karşılaştırması + GridSearchCV
  ✓ StratifiedKFold (5 kat)
  ✓ class_weight="balanced"
  ✓ Threshold optimizasyonu (F1 bazlı)
  ✓ Precision / Recall / F1 dengesi
  ✓ Confusion matrix + karşılaştırma grafiği
"""

import os, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.svm             import SVC
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics         import (classification_report, confusion_matrix,
                                      f1_score, accuracy_score,
                                      precision_score, recall_score,
                                      roc_curve, auc, precision_recall_curve)
from sklearn.base            import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore")


class PosBenzerlikSiniflandirici(BaseEstimator, ClassifierMixin):
    """
    Sadece pozitif örneklerin ortalama vektörüne (centroid) kosinüs benzerliği
    hesaplar. Negatif örneklere hiç bakmaz — 'Hey Pakize nasıl duyulur'
    öğrenir, bilinmeyen kelimeler otomatik negatif kalır.
    """

    def __init__(self, std_agirlik=0.3):
        # std_agirlik: centroid benzerliğine ek olarak
        # her özniteliğin tutarlılığını (düşük std → güvenilir) ağırlıklandırır
        self.std_agirlik = std_agirlik

    def fit(self, X, y):
        X_pos = X[y == 1]
        self.centroid_  = X_pos.mean(axis=0)
        self.std_       = X_pos.std(axis=0) + 1e-8   # sıfıra bölmeyi önle
        # Normalize edilmiş centroid (kosinüs için)
        norm = np.linalg.norm(self.centroid_)
        self.centroid_norm_ = self.centroid_ / norm if norm > 0 else self.centroid_
        self.classes_ = np.array([0, 1])
        return self

    def _benzerlik(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_norm = X / norms
        cos_sim = X_norm @ self.centroid_norm_          # [-1, 1]
        cos_prob = (cos_sim + 1) / 2                    # [0, 1]

        # Standart sapma ağırlığı: centroid'e yakın olan öznitelikler daha güvenilir
        onem = 1.0 / (self.std_ + 1e-8)
        onem = onem / onem.sum()
        mahal = np.exp(-np.sum(((X - self.centroid_) * onem) ** 2, axis=1) * self.std_agirlik)

        return np.clip(cos_prob * (0.7 + 0.3 * mahal), 0, 1)

    def predict_proba(self, X):
        p = self._benzerlik(X)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

BASE_DIR      = Path(__file__).resolve().parent.parent
CSV_YOLU      = str(BASE_DIR / "data" / "processed" / "proje_veriseti.csv")
MODEL_KLASOR  = str(BASE_DIR / "models" / "improved")
GORSEL_KLASOR = str(BASE_DIR / "results" / "visualizations")

os.makedirs(MODEL_KLASOR,  exist_ok=True)
os.makedirs(GORSEL_KLASOR, exist_ok=True)


def veri_yukle():
    if not os.path.exists(CSV_YOLU):
        raise FileNotFoundError(f"'{CSV_YOLU}' bulunamadı. Önce oznitelik_cikarma.py çalıştır!")
    df = pd.read_csv(CSV_YOLU)
    X  = df.drop(columns=["dosya", "etiket"]).values.astype(np.float32)
    y  = df["etiket"].values
    print(f"Veri: {X.shape[0]} örnek, {X.shape[1]} öznitelik")
    print(f"  Pozitif: {(y==1).sum()}  |  Negatif: {(y==0).sum()}\n")
    return X, y


def model_listesi():
    return {
        "Pos Benzerlik": (
            PosBenzerlikSiniflandirici(),
            {"std_agirlik": [0.1, 0.3, 0.5]}
        ),
        "SVM (RBF)": (
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
            {"C": [1, 10, 100], "gamma": ["scale", "auto"]}
        ),
        "SVM (Linear)": (
            SVC(kernel="linear", probability=True, class_weight="balanced"),
            {"C": [0.1, 1, 10]}
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20]}
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        ),
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            {"C": [0.1, 1, 10]}
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
        ),
    }


def en_iyi_esik(model, X_val, y_val):
    """F1 bazlı en iyi threshold."""
    proba = model.predict_proba(X_val)[:, 1]
    en_iyi_f1, en_iyi_t = 0, 0.5
    for t in np.arange(0.25, 0.96, 0.05):
        yp = (proba >= t).astype(int)
        f1 = f1_score(y_val, yp, zero_division=0)
        if f1 > en_iyi_f1:
            en_iyi_f1, en_iyi_t = f1, t
    return round(float(en_iyi_t), 2)


def egit(X_tr, X_val, X_te, y_tr, y_val, y_te):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sonuclar = {}

    for isim, (model, params) in model_listesi().items():
        print(f"  ► {isim}...", end=" ", flush=True)
        try:
            gs = GridSearchCV(model, params, cv=cv,
                              scoring="f1", n_jobs=-1, refit=True)
            gs.fit(X_tr, y_tr)
            esik  = en_iyi_esik(gs.best_estimator_, X_val, y_val)
            proba = gs.predict_proba(X_te)[:, 1]
            yp    = (proba >= esik).astype(int)
            sonuclar[isim] = {
                "model":     gs.best_estimator_,
                "params":    gs.best_params_,
                "cv_f1":     round(float(gs.best_score_), 4),
                "esik":      esik,
                "proba":     proba,
                "y_pred":    yp,
                "accuracy":  round(accuracy_score(y_te, yp),              4),
                "f1":        round(f1_score(y_te, yp,        zero_division=0), 4),
                "precision": round(precision_score(y_te, yp, zero_division=0), 4),
                "recall":    round(recall_score(y_te, yp,    zero_division=0), 4),
            }
            r = sonuclar[isim]
            print(f"Acc={r['accuracy']:.2f}  Prec={r['precision']:.2f}  "
                  f"Rec={r['recall']:.2f}  F1={r['f1']:.2f}  Eşik={esik}")
        except Exception as e:
            print(f"HATA: {e}")

    return sonuclar


def gorsel_kaydet(y_te, en_iyi_adi, sonuclar):
    # Eski PNG'leri temizle
    for eski in Path(GORSEL_KLASOR).glob("*.png"):
        eski.unlink()

    isimler = list(sonuclar.keys())
    renkler = ["#4A90D9", "#E8593C", "#EF9F27", "#2ecc71",
               "#9B59B6", "#1ABC9C", "#E74C3C"][:len(isimler)]

    # ── 1. Confusion Matrix (sayı + yüzde) ──────────────────
    yp = sonuclar[en_iyi_adi]["y_pred"]
    cm = confusion_matrix(y_te, yp)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negatif", "Pozitif"])
    ax.set_yticklabels(["Negatif", "Pozitif"])
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
    ax.set_title(f"Confusion Matrix — {en_iyi_adi}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax); plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_KLASOR, "confusion_matrix.png"), dpi=120)
    plt.close()

    # ── 2. Model Karşılaştırma (Acc/Prec/Rec/F1/CV-F1) ─────
    x = np.arange(len(isimler)); gn = 0.15
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 2*gn, [sonuclar[m]["accuracy"]  for m in isimler], gn, label="Accuracy",  color="#4A90D9")
    ax.bar(x - 1*gn, [sonuclar[m]["precision"] for m in isimler], gn, label="Precision", color="#E8593C")
    ax.bar(x,         [sonuclar[m]["recall"]    for m in isimler], gn, label="Recall",    color="#EF9F27")
    ax.bar(x + 1*gn, [sonuclar[m]["f1"]        for m in isimler], gn, label="Test F1",   color="#2ecc71")
    ax.bar(x + 2*gn, [sonuclar[m]["cv_f1"]     for m in isimler], gn, label="CV F1",     color="#9B59B6")
    ax.set_xticks(x); ax.set_xticklabels(isimler, rotation=20, ha="right")
    ax.set_ylim(0, 1.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right"); ax.set_title("Model Karşılaştırması (Test + CV F1)")
    plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_KLASOR, "model_comparison.png"), dpi=120)
    plt.close()

    # ── 3. ROC Eğrileri ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for isim, renk in zip(isimler, renkler):
        try:
            fpr, tpr, _ = roc_curve(y_te, sonuclar[isim]["proba"])
            auc_val = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=renk, lw=2,
                    label=f"{isim}  (AUC={auc_val:.3f})")
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR / Recall)")
    ax.set_title("ROC Eğrileri — Tüm Modeller")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_KLASOR, "roc_curves.png"), dpi=120)
    plt.close()

    # ── 4. Precision-Recall Eğrileri ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for isim, renk in zip(isimler, renkler):
        try:
            prec, rec, _ = precision_recall_curve(y_te, sonuclar[isim]["proba"])
            ap = float(np.trapz(prec[::-1], rec[::-1]))
            ax.plot(rec, prec, color=renk, lw=2,
                    label=f"{isim}  (AP={ap:.3f})")
        except Exception:
            pass
    baseline = float((y_te == 1).sum()) / len(y_te)
    ax.axhline(baseline, color="gray", linestyle="--", lw=1,
               label=f"Baseline ({baseline:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Eğrileri — Tüm Modeller")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_KLASOR, "pr_curves.png"), dpi=120)
    plt.close()

    # ── 5. Performans Özet Tablosu ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("off")
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "CV F1", "Eşik"]
    satirlar = []
    for isim in isimler:
        r = sonuclar[isim]
        satirlar.append([
            isim,
            f"{r['accuracy']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['cv_f1']:.4f}",
            f"{r['esik']:.2f}",
        ])
    tablo = ax.table(cellText=satirlar, colLabels=cols,
                     loc="center", cellLoc="center")
    tablo.auto_set_font_size(False)
    tablo.set_fontsize(9)
    tablo.scale(1, 1.6)
    for j in range(len(cols)):
        tablo[(0, j)].set_facecolor("#2a2a3e")
        tablo[(0, j)].set_text_props(color="white", fontweight="bold")
    for i, isim in enumerate(isimler):
        if isim == en_iyi_adi:
            for j in range(len(cols)):
                tablo[(i + 1, j)].set_facecolor("#d4edda")
    ax.set_title("Model Performans Özeti  (★ yeşil = En İyi CV F1)", pad=14)
    plt.tight_layout()
    plt.savefig(os.path.join(GORSEL_KLASOR, "performans_tablosu.png"), dpi=120)
    plt.close()

    olusturulanlar = [
        "confusion_matrix.png", "model_comparison.png",
        "roc_curves.png", "pr_curves.png", "performans_tablosu.png",
    ]
    print(f"  Görseller ({GORSEL_KLASOR}):")
    for g_ad in olusturulanlar:
        print(f"    ✓ {g_ad}")


def main():
    print("=" * 60)
    print("  HEY PAKİZE — Model Eğitimi (Final)")
    print("=" * 60 + "\n")

    X, y = veri_yukle()

    # %70 train / %15 val / %15 test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    print(f"Train: {len(y_tr)}  Val: {len(y_val)}  Test: {len(y_te)}\n")

    sc = StandardScaler()
    X_tr  = sc.fit_transform(X_tr)
    X_val = sc.transform(X_val)
    X_te  = sc.transform(X_te)

    print("Modeller eğitiliyor...\n")
    sonuclar = egit(X_tr, X_val, X_te, y_tr, y_val, y_te)

    # En iyi model — CV F1 bazlı (test seti sadece raporlama için)
    en_iyi_adi = max(sonuclar, key=lambda m: sonuclar[m]["cv_f1"])
    en_iyi     = sonuclar[en_iyi_adi]

    print(f"\n{'═'*60}")
    print(f"  EN İYİ MODEL  : {en_iyi_adi}")
    print(f"  Accuracy      : {en_iyi['accuracy']}")
    print(f"  Precision     : {en_iyi['precision']}")
    print(f"  Recall        : {en_iyi['recall']}")
    print(f"  F1-Score      : {en_iyi['f1']}")
    print(f"  Eşik          : {en_iyi['esik']}")
    print(f"{'═'*60}\n")
    print(classification_report(y_te, en_iyi["y_pred"],
                                  target_names=["Negatif","Pozitif"]))

    # Kaydet
    joblib.dump(en_iyi["model"], os.path.join(MODEL_KLASOR, "best_model.pkl"))
    joblib.dump(sc,              os.path.join(MODEL_KLASOR, "scaler.pkl"))

    meta = {
        "model_adi":   en_iyi_adi,
        "esik":        en_iyi["esik"],
        "accuracy":    en_iyi["accuracy"],
        "precision":   en_iyi["precision"],
        "recall":      en_iyi["recall"],
        "f1":          en_iyi["f1"],
        "best_params": en_iyi["params"],
    }
    with open(os.path.join(MODEL_KLASOR, "model_metadata.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Görseller oluşturuluyor...")
    gorsel_kaydet(y_te, en_iyi_adi, sonuclar)
    print(f"\nModel → {MODEL_KLASOR}/best_model.pkl")
    print("Tamamlandı! 🎉")


if __name__ == "__main__":
    main()