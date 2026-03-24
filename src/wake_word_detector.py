"""
wake_word_detector.py
---------------------
"Hey Pakize" gerçek zamanlı tespiti.

Kullanım:
  python wake_word_detector.py           # terminal
  python wake_word_detector.py --gui     # GUI
"""

import argparse, os, queue, threading, time, warnings, json
from collections import deque
from pathlib import Path
import numpy as np
import librosa
import joblib
import sounddevice as sd
warnings.filterwarnings("ignore")

SR          = 16000
SURE        = 1.5
N_MFCC      = 13
HOP_LENGTH  = 512
N_FFT       = 2048
MAX_SAMPLES = int(SR * SURE)
BLOK_SURE   = 0.5
SMOOTH_N    = 3   # son kaç tahminin ortalaması alınır
MODEL_KLASOR = str(Path(__file__).resolve().parent.parent / "models" / "improved")


def model_yukle():
    for dosya in ["best_model.pkl", "scaler.pkl"]:
        if not os.path.exists(os.path.join(MODEL_KLASOR, dosya)):
            raise FileNotFoundError(
                f"'{dosya}' bulunamadı. Önce model_egitimi.py çalıştır!")
    model  = joblib.load(os.path.join(MODEL_KLASOR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_KLASOR, "scaler.pkl"))
    meta   = {}
    mp = os.path.join(MODEL_KLASOR, "model_metadata.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    return model, scaler, meta


def pre_emphasis(y, coef=0.97):
    return np.append(y[0], y[1:] - coef * y[:-1])


def oznitelik_cikart(y_ham):
    """Ham tampon → 26 boyutlu öznitelik (MFCC + Delta)."""
    rms_ham = float(np.sqrt(np.mean(y_ham ** 2)))
    if rms_ham < 0.005:
        return None   # Sessizlik — tahmin yapma

    y = pre_emphasis(y_ham)
    y_tr, _ = librosa.effects.trim(y, top_db=30)
    if len(y_tr) < SR * 0.3:
        y_tr = y
    peak = np.max(np.abs(y_tr))
    if peak > 0:
        y_tr = y_tr / peak

    if len(y_tr) >= MAX_SAMPLES:
        start = (len(y_tr) - MAX_SAMPLES) // 2
        y_tr = y_tr[start:start + MAX_SAMPLES]
    else:
        pad = MAX_SAMPLES - len(y_tr)
        y_tr = np.pad(y_tr, (pad // 2, pad - pad // 2))

    mfcc   = librosa.feature.mfcc(y=y_tr, sr=SR, n_mfcc=N_MFCC,
                                    hop_length=HOP_LENGTH, n_fft=N_FFT)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc,   axis=1),
        np.mean(delta,  axis=1),
        np.mean(delta2, axis=1),
    ]).reshape(1, -1).astype(np.float32)


def tahmin_et(y_ham, model, scaler):
    of = oznitelik_cikart(y_ham)
    if of is None:
        return None
    return float(model.predict_proba(scaler.transform(of))[0][1])


# ══════════════════════════════════════════
# TERMINAL MODU
# ══════════════════════════════════════════

def terminal_modu(esik):
    model, scaler, meta = model_yukle()
    print(f"Model: {meta.get('model_adi','?')}  F1={meta.get('f1','?')}  Eşik={esik}")
    print("\n🎙️  Dinleniyor... (Ctrl+C ile dur)\n")

    q       = queue.Queue()
    tampon  = np.zeros(MAX_SAMPLES, dtype=np.float32)
    gecmis  = deque(maxlen=SMOOTH_N)

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1,
                        blocksize=int(SR * BLOK_SURE), callback=cb):
        while True:
            try:
                blok = q.get(timeout=2.0)
            except queue.Empty:
                continue
            tampon = np.roll(tampon, -len(blok))
            tampon[-len(blok):] = blok

            p_ham = tahmin_et(tampon, model, scaler)
            if p_ham is None:
                p_ham = 0.0
            gecmis.append(p_ham)
            p = sum(gecmis) / len(gecmis)

            bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
            if p >= esik:
                print(f"\r  ✅ HEY PAKİZE! [{bar}] {p:.2f}  ")
                print()
            else:
                print(f"\r  🎙  [{bar}] {p:.2f}  ", end="")


# ══════════════════════════════════════════
# GUI MODU
# ══════════════════════════════════════════

def gui_modu(esik):
    try:
        import tkinter as tk
        from tkinter import ttk, scrolledtext
    except ImportError:
        print("tkinter yok.")
        return

    model, scaler, meta = model_yukle()

    pencere = tk.Tk()
    pencere.title("Hey Pakize — Wake Word Detector")
    pencere.geometry("520x560")
    pencere.resizable(False, False)
    pencere.configure(bg="#1e1e2e")

    C = {
        "bg": "#1e1e2e", "panel": "#2a2a3e", "acik": "#313149",
        "yesil": "#4CAF50", "kirmizi": "#E8593C", "sari": "#EF9F27",
        "beyaz": "#cdd6f4", "gri": "#888899", "mavi": "#89b4fa",
    }

    tk.Label(pencere, text="🎙  Hey Pakize",
             font=("Helvetica", 20, "bold"),
             bg=C["bg"], fg=C["beyaz"]).pack(pady=(16, 2))
    tk.Label(pencere, text="Wake Word Detection System",
             font=("Helvetica", 10), bg=C["bg"], fg=C["gri"]).pack()

    durum = tk.Label(pencere, text="⏹  Durduruldu",
                     font=("Helvetica", 14, "bold"),
                     bg=C["panel"], fg=C["gri"])
    durum.pack(fill="x", padx=20, pady=12, ipady=12)

    def bar_satir(etiket, renk):
        f = tk.Frame(pencere, bg=C["bg"])
        f.pack(fill="x", padx=20, pady=3)
        tk.Label(f, text=etiket, bg=C["bg"], fg=C["gri"],
                 font=("Helvetica", 9), width=12, anchor="w").pack(side="left")
        dv = tk.DoubleVar()
        ttk.Progressbar(f, variable=dv, maximum=1.0, length=320).pack(side="left", padx=6)
        yazi = tk.Label(f, text="0.00", bg=C["bg"], fg=renk,
                        font=("Helvetica", 10, "bold"), width=5)
        yazi.pack(side="left")
        return dv, yazi

    guven_dv, guven_yazi = bar_satir("Güven:", C["mavi"])
    ses_dv,   _          = bar_satir("Ses:", C["gri"])

    f = tk.Frame(pencere, bg=C["bg"])
    f.pack(fill="x", padx=20, pady=3)
    tk.Label(f, text="Eşik:", bg=C["bg"], fg=C["gri"],
             font=("Helvetica", 9), width=12, anchor="w").pack(side="left")
    esik_dv = tk.DoubleVar(value=esik)
    tk.Scale(f, from_=0.3, to=0.95, resolution=0.05,
             orient="horizontal", variable=esik_dv, length=270,
             bg=C["bg"], fg=C["beyaz"], troughcolor=C["acik"],
             highlightthickness=0).pack(side="left", padx=6)
    tk.Label(f, textvariable=esik_dv, bg=C["bg"], fg=C["sari"],
             font=("Helvetica", 10, "bold"), width=4).pack(side="left")

    tk.Label(pencere, text="Tespit geçmişi", bg=C["bg"],
             fg=C["gri"], font=("Helvetica", 9)).pack(anchor="w", padx=22, pady=(10, 0))
    log = scrolledtext.ScrolledText(pencere, height=9, state="disabled",
                                     bg=C["panel"], fg=C["beyaz"],
                                     font=("Courier", 10), bd=0)
    log.pack(fill="x", padx=20, pady=(2, 10))

    def log_yaz(msg):
        log.configure(state="normal")
        log.insert("end", msg + "\n")
        log.see("end")
        log.configure(state="disabled")

    aktif      = {"v": False}
    son_tespit = {"t": 0}
    q          = queue.Queue()

    def guncelle(p, rms, e):
        guven_dv.set(p)
        guven_yazi.config(text=f"{p:.2f}")
        ses_dv.set(min(rms * 10, 1.0))
        if p >= e:
            durum.config(text="✅  HEY PAKİZE!", fg=C["yesil"])
        else:
            durum.config(text="🎙  Dinleniyor...", fg=C["mavi"])

    def tespit_goster(p):
        z = time.strftime("%H:%M:%S")
        durum.config(text="✅  HEY PAKİZE!", fg=C["yesil"])
        log_yaz(f"[{z}] ✅ HEY PAKİZE! (güven: {p:.2f})")
        pencere.after(2000, lambda: durum.config(
            text="🎙  Dinleniyor...", fg=C["mavi"]))

    def dongu():
        tampon = np.zeros(MAX_SAMPLES, dtype=np.float32)
        gecmis = deque(maxlen=SMOOTH_N)

        def cb(indata, frames, t, status):
            q.put(indata[:, 0].copy())

        with sd.InputStream(samplerate=SR, channels=1,
                            blocksize=int(SR * BLOK_SURE), callback=cb):
            while aktif["v"]:
                try:
                    blok = q.get(timeout=1.0)
                except queue.Empty:
                    continue
                tampon_yeni = np.roll(tampon, -len(blok))
                tampon_yeni[-len(blok):] = blok
                tampon[:] = tampon_yeni

                e   = esik_dv.get()
                rms = float(np.sqrt(np.mean(tampon ** 2)))
                p_ham = tahmin_et(tampon, model, scaler)
                if p_ham is None:
                    p_ham = 0.0
                gecmis.append(p_ham)
                p = sum(gecmis) / len(gecmis)

                pencere.after(0, lambda _p=p, _r=rms, _e=e: guncelle(_p, _r, _e))

                if p >= e and time.time() - son_tespit["t"] > 2.0:
                    son_tespit["t"] = time.time()
                    pencere.after(0, lambda _p=p: tespit_goster(_p))

    def baslat_durdur():
        if not aktif["v"]:
            aktif["v"] = True
            dugme.config(text="⏹  Durdur", bg=C["kirmizi"])
            log_yaz(f"[{time.strftime('%H:%M:%S')}] Dinleme başladı. Eşik={esik_dv.get():.2f}")
            threading.Thread(target=dongu, daemon=True).start()
        else:
            aktif["v"] = False
            dugme.config(text="▶  Başlat", bg=C["yesil"])
            durum.config(text="⏹  Durduruldu", fg=C["gri"])
            log_yaz(f"[{time.strftime('%H:%M:%S')}] Durduruldu.")

    dugme = tk.Button(pencere, text="▶  Başlat",
                      font=("Helvetica", 13, "bold"),
                      bg=C["yesil"], fg="white",
                      bd=0, padx=20, pady=8, cursor="hand2",
                      command=baslat_durdur)
    dugme.pack(pady=(0, 10))

    if meta:
        bilgi = (f"Model: {meta.get('model_adi','?')}  |  "
                 f"F1={meta.get('f1','?')}  |  "
                 f"Acc={meta.get('accuracy','?')}")
        tk.Label(pencere, text=bilgi, bg=C["bg"], fg=C["gri"],
                 font=("Helvetica", 8)).pack()

    pencere.mainloop()


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("-t", "--threshold", type=float, default=None)
    args = ap.parse_args()

    esik = args.threshold
    if esik is None:
        mp = os.path.join(MODEL_KLASOR, "model_metadata.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                esik = json.load(f).get("esik", 0.65)
        else:
            esik = 0.65

    try:
        if args.gui:
            gui_modu(esik)
        else:
            terminal_modu(esik)
    except KeyboardInterrupt:
        print("\nDurduruldu.")
    except FileNotFoundError as e:
        print(f"\nHATA: {e}")
