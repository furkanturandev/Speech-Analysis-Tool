"""
COE216 - Signals and Systems: Speech vs Silence & Voiced vs Unvoiced Analysis
==============================================================================
Modern GUI versiyonu — Tkinter ile dosya seçme + matplotlib dark-theme grafik.

Kullanim:
  python speech_analysis.py
  → Açılan pencereden .wav dosyanızı seçin.

Gereksinimler: numpy, scipy, matplotlib  (librosa GEREKMEZ)
"""

import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import wave

# matplotlib backend — Tkinter ile gömülü çalışması için
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

from scipy.signal import medfilt


# ═════════════════════════════════════════════════════════════════════════════
#  RENK PALETİ — Modern Dark Theme
# ═════════════════════════════════════════════════════════════════════════════
C = {
    "bg":       "#1e1e2e",   "surface":  "#282840",   "card":     "#313150",
    "accent":   "#7c3aed",   "acc_hov":  "#6d28d9",   "text":     "#e2e8f0",
    "dim":      "#94a3b8",   "green":    "#22c55e",   "yellow":   "#eab308",
    "red":      "#ef4444",   "blue":     "#3b82f6",   "orange":   "#f97316",
    "plot_bg":  "#1a1a2e",   "grid":     "#374151",
}


# ═════════════════════════════════════════════════════════════════════════════
#  YAPILANDIRMA PARAMETRELERİ
# ═════════════════════════════════════════════════════════════════════════════
FRAME_MS        = 20       # Pencere uzunluğu (ms)
OVERLAP_RATIO   = 0.50     # %50 örtüşme
NOISE_DUR_MS    = 200      # İlk 200 ms → gürültü referansı
ENERGY_MULT     = 3.0      # Dinamik eşik çarpanı
HANGOVER_FRAMES = 4        # Hangover pencere sayısı
MEDIAN_KERNEL   = 7        # Median filtre boyutu (tek sayı)
ZCR_THRESHOLD   = 0.10     # Voiced/Unvoiced ZCR sınırı


# ═════════════════════════════════════════════════════════════════════════════
#  1) SİNYAL HAZIRLAMA — wave modülü ile (librosa gereksiz)
#     .wav dosyasını okur, mono yapar, [-1,1] normalize eder
# ═════════════════════════════════════════════════════════════════════════════
def load_wav(wav_path: str):
    """wave modülü ile WAV okur → mono float64 [-1,1] sinyal + fs döndürür."""
    with wave.open(wav_path, 'r') as wf:
        n_ch    = wf.getnchannels()
        sw      = wf.getsampwidth()       # byte / örnek
        fs      = wf.getframerate()
        n_frames = wf.getnframes()
        raw     = wf.readframes(n_frames)

    # Byte → numpy
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
    if dt is None:
        raise ValueError(f"Desteklenmeyen sample width: {sw}")
    sig = np.frombuffer(raw, dtype=dt).astype(np.float64)

    # Stereo / çok kanallı → Mono
    # Toplam örnek sayısını kanal sayısına göre kırp (artık byte varsa)
    if n_ch >= 2:
        usable = len(sig) - (len(sig) % n_ch)   # kanal sayısına tam bölünen kısım
        sig = sig[:usable].reshape(-1, n_ch).mean(axis=1)

    # Normalize [-1, 1]
    mx = np.max(np.abs(sig))
    if mx > 0:
        sig /= mx
    return sig, fs


# ═════════════════════════════════════════════════════════════════════════════
#  2) PENCERELEME — 20 ms Hamming, %50 overlap
# ═════════════════════════════════════════════════════════════════════════════
def frame_signal(signal, fs):
    """Sinyali Hamming pencereli çerçevelere böler."""
    frame_len = int(fs * FRAME_MS / 1000)
    hop_len   = int(frame_len * (1 - OVERLAP_RATIO))
    n_frames  = max(1, 1 + (len(signal) - frame_len) // hop_len)

    padded_len = frame_len + (n_frames - 1) * hop_len
    pad = np.zeros(padded_len)
    copy_len = min(len(signal), padded_len)   # taşmayı önle
    pad[:copy_len] = signal[:copy_len]

    idx    = np.arange(frame_len)[None, :] + np.arange(n_frames)[:, None] * hop_len
    frames = pad[idx] * np.hamming(frame_len)[None, :]
    return frames, frame_len, hop_len


# ═════════════════════════════════════════════════════════════════════════════
#  3) VAD — Karesel Enerji + Dinamik Eşik + Hangover + Median
# ═════════════════════════════════════════════════════════════════════════════
def squared_energy(frames):
    """E[n] = (1/N) Σ x²  — karesel enerji."""
    return np.mean(frames ** 2, axis=1)


def zero_crossing_rate(frames):
    """ZCR = sıfır geçiş oranı."""
    s = np.sign(frames)
    return np.sum(np.abs(np.diff(s, axis=1)), axis=1) / (2 * (frames.shape[1] - 1))


def adaptive_threshold(energy, fs, hop_len):
    """İlk 200 ms gürültü → eşik = μ + k·σ."""
    nf = max(int((NOISE_DUR_MS / 1000) * fs / hop_len), 1)
    ne = energy[:nf]
    return np.mean(ne) + ENERGY_MULT * np.std(ne)


def hangover(mask):
    """Konuşma→sessizlik geçişinde HANGOVER_FRAMES kadar 1 tutar."""
    r = mask.copy(); cnt = 0
    for i in range(len(r)):
        if r[i] == 1:
            cnt = HANGOVER_FRAMES
        elif cnt > 0:
            r[i] = 1; cnt -= 1
    return r


def run_analysis(wav_path):
    """Tüm analiz → signal, fs, energy, zcr, labels, frame_len, hop_len, stats."""
    sig, fs = load_wav(wav_path)
    frames, fl, hl = frame_signal(sig, fs)

    energy = squared_energy(frames)
    thr    = adaptive_threshold(energy, fs, hl)
    mask   = (energy >= thr).astype(np.int32)
    mask   = hangover(mask)
    mask   = medfilt(mask, kernel_size=MEDIAN_KERNEL).astype(np.int32)

    zcr    = zero_crossing_rate(frames)
    labels = np.zeros(len(frames), dtype=np.int32)
    for i in range(len(frames)):
        if mask[i] == 1:
            labels[i] = 2 if zcr[i] < ZCR_THRESHOLD else 1

    nv = int(np.sum(labels == 2))
    nu = int(np.sum(labels == 1))
    ns = int(np.sum(labels == 0))
    nt = len(labels)
    stats = dict(
        fs=fs, dur=len(sig)/fs, n_samp=len(sig), n_fr=nt,
        nv=nv, nu=nu, ns=ns, thr=thr,
        pv=100*nv/nt, pu=100*nu/nt, ps=100*ns/nt,
    )
    return sig, fs, energy, zcr, labels, fl, hl, stats


# ═════════════════════════════════════════════════════════════════════════════
#  5) SPEECH BÖLGELERINI YENİ WAV OLARAK KAYDET
#     VAD maskesinde speech (labels != 0) olan pencereleri uç uca ekler,
#     orijinal örnekleme hızı ve 16-bit PCM olarak yeni .wav yazar.
# ═════════════════════════════════════════════════════════════════════════════
def save_speech_wav(wav_path, output_path, signal, fs, labels, frame_len, hop_len):
    """
    VAD tarafından 'Speech' (Voiced veya Unvoiced) olarak etiketlenen
    pencerelere karşılık gelen orijinal sinyal parçalarını uç uca ekleyerek
    yeni bir .wav dosyası olarak kaydeder.

    Parametreler:
        wav_path    : Orijinal dosya yolu (bilgi amaçlı)
        output_path : Kaydedilecek yeni dosya yolu
        signal      : Normalize edilmiş orijinal sinyal (1-B numpy dizisi)
        fs          : Örnekleme hızı (Hz)
        labels      : Pencere etiketleri (0=Sessizlik, 1=Unvoiced, 2=Voiced)
        frame_len   : Pencere uzunluğu (örnek sayısı)
        hop_len     : Kayma (hop) uzunluğu (örnek sayısı)

    Döndürür:
        info : dict — orijinal süre, yeni süre, temizlenen oran
    """
    # Orijinal (Hamming uygulanmamış) sinyal parçalarını topla
    speech_chunks = []
    n_frames = len(labels)

    for i in range(n_frames):
        if labels[i] != 0:  # Voiced (2) veya Unvoiced (1) → speech
            start = i * hop_len
            end   = min(start + frame_len, len(signal))
            speech_chunks.append(signal[start:end])

    # Overlap'li pencereleri birleştirirken çakışan bölgeleri düzgün ekle
    # Basit yöntem: overlap-add ile birleştir
    if not speech_chunks:
        # Hiç konuşma bulunamadı — boş dosya
        speech_signal = np.array([], dtype=np.float64)
    else:
        # Her speech penceresinin başlangıç indekslerini ve bitişiklik durumunu bul
        # Bitişik pencereler → overlap-add; ayrık bölgeler → uç uca ekle
        segments = []       # [(start_frame, end_frame), ...]  bitişik gruplar
        seg_start = None

        for i in range(n_frames):
            if labels[i] != 0:
                if seg_start is None:
                    seg_start = i
                seg_end = i
            else:
                if seg_start is not None:
                    segments.append((seg_start, seg_end))
                    seg_start = None
        if seg_start is not None:
            segments.append((seg_start, seg_end))

        # Her bitişik segmenti orijinal sinyalden kes ve uç uca ekle
        result_parts = []
        for (sf, ef) in segments:
            sample_start = sf * hop_len
            sample_end   = min(ef * hop_len + frame_len, len(signal))
            result_parts.append(signal[sample_start:sample_end])

        speech_signal = np.concatenate(result_parts)

    # Süre hesapları
    orig_dur = len(signal) / fs
    new_dur  = len(speech_signal) / fs
    removed  = orig_dur - new_dur
    pct      = (removed / orig_dur * 100) if orig_dur > 0 else 0

    # 16-bit PCM WAV olarak kaydet
    speech_int16 = np.clip(speech_signal * 32767, -32768, 32767).astype(np.int16)
    with wave.open(output_path, 'w') as wf:
        wf.setnchannels(1)          # Mono (analiz mono yapıldı)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(fs)         # Orijinal örnekleme hızı korunur
        wf.writeframes(speech_int16.tobytes())

    info = {
        "orig_dur": orig_dur,
        "new_dur": new_dur,
        "removed_dur": removed,
        "pct_removed": pct,
        "output_path": output_path,
    }
    return info


# ═════════════════════════════════════════════════════════════════════════════
#  MODERN GUI
# ═════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("COE216 - Speech Analysis Tool")
        self.root.configure(bg=C["bg"])
        self.root.state("zoomed")
        self.root.minsize(1100, 700)
        self._build()

    # ─── UI İNŞA ─────────────────────────────────────────────────────────
    def _build(self):
        # ── ÜST BAR ──
        top = tk.Frame(self.root, bg=C["surface"], height=70)
        top.pack(fill="x"); top.pack_propagate(False)

        lf = tk.Frame(top, bg=C["surface"])
        lf.pack(side="left", padx=20, pady=10)
        tk.Label(lf, text="🎙 Speech Analysis Tool", bg=C["surface"],
                 fg=C["text"], font=("Segoe UI", 16, "bold")).pack(anchor="w")
        tk.Label(lf, text="COE216 - Signals and Systems", bg=C["surface"],
                 fg=C["dim"], font=("Segoe UI", 9)).pack(anchor="w")

        rf = tk.Frame(top, bg=C["surface"])
        rf.pack(side="right", padx=20, pady=15)

        self.btn_save = tk.Button(
            rf, text="💾  PNG Kaydet", font=("Segoe UI", 10, "bold"),
            bg="#374151", fg=C["text"], activebackground="#4b5563",
            bd=0, padx=16, pady=8, cursor="hand2",
            command=self._save, state="disabled")
        self.btn_save.pack(side="right", padx=(8, 0))

        self.btn_export = tk.Button(
            rf, text="🔊  Konuşmayı Kaydet", font=("Segoe UI", 10, "bold"),
            bg="#065f46", fg=C["text"], activebackground="#047857",
            bd=0, padx=16, pady=8, cursor="hand2",
            command=self._export_speech, state="disabled")
        self.btn_export.pack(side="right", padx=(8, 0))

        tk.Button(
            rf, text="📂  WAV Dosyası Seç", font=("Segoe UI", 10, "bold"),
            bg=C["accent"], fg="white", activebackground=C["acc_hov"],
            bd=0, padx=20, pady=8, cursor="hand2",
            command=self._open).pack(side="right")

        self.file_lbl = tk.Label(rf, text="Henüz dosya seçilmedi",
                                 bg=C["surface"], fg=C["dim"],
                                 font=("Segoe UI", 9))
        self.file_lbl.pack(side="right", padx=(0, 16))

        # ── İSTATİSTİK KARTLARI ──
        sf = tk.Frame(self.root, bg=C["bg"])
        sf.pack(fill="x", padx=16, pady=(12, 4))
        self._cards = {}
        for key, icon_lbl in [("dur","⏱ Süre"), ("fs","📊 Örnekleme"),
                               ("nfr","🔲 Pencere"), ("pv","🟢 Voiced"),
                               ("pu","🟡 Unvoiced"), ("ps","⬜ Sessizlik")]:
            c = tk.Frame(sf, bg=C["card"], highlightthickness=1,
                         highlightbackground="#3f3f5f")
            c.pack(side="left", fill="both", expand=True, padx=4)
            inner = tk.Frame(c, bg=C["card"]); inner.pack(padx=14, pady=10)
            vl = tk.Label(inner, text="—", bg=C["card"], fg=C["accent"],
                          font=("Consolas", 17, "bold")); vl.pack()
            tk.Label(inner, text=icon_lbl, bg=C["card"], fg=C["dim"],
                     font=("Segoe UI", 9)).pack()
            self._cards[key] = vl

        # ── GRAFİK ALANI ──
        self.plot_area = tk.Frame(self.root, bg=C["bg"])
        self.plot_area.pack(fill="both", expand=True, padx=16, pady=(4, 8))
        self._welcome()

        # ── ALT BAR ──
        bb = tk.Frame(self.root, bg=C["surface"], height=30)
        bb.pack(fill="x", side="bottom"); bb.pack_propagate(False)
        self.status = tk.Label(bb, text="  Hazır — Bir WAV dosyası seçerek başlayın",
                               bg=C["surface"], fg=C["dim"],
                               font=("Segoe UI", 9), anchor="w")
        self.status.pack(fill="x", padx=8, pady=5)

    # ─── HOŞ GELDİN ─────────────────────────────────────────────────────
    def _welcome(self):
        for w in self.plot_area.winfo_children(): w.destroy()
        f = tk.Frame(self.plot_area, bg=C["bg"])
        f.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(f, text="🎵", bg=C["bg"], font=("Segoe UI", 48)).pack(pady=(0,10))
        tk.Label(f, text="WAV Dosyanızı Seçin", bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 18, "bold")).pack()
        tk.Label(f, text='Yukarıdaki "📂  WAV Dosyası Seç" butonuna tıklayın.',
                 bg=C["bg"], fg=C["dim"], font=("Segoe UI", 11),
                 justify="center").pack(pady=(8,0))

    # ─── DOSYA AÇ ───────────────────────────────────────────────────────
    def _open(self):
        fp = filedialog.askopenfilename(
            title="WAV Dosyası Seçin",
            filetypes=[("WAV", "*.wav"), ("Tümü", "*.*")],
            initialdir=os.getcwd())
        if not fp: return

        self.file_lbl.config(text=f"📄 {os.path.basename(fp)}")
        self.status.config(text=f"  Analiz ediliyor: {os.path.basename(fp)} ...")
        self.root.update_idletasks()

        try:
            sig, fs, energy, zcr, labels, fl, hl, st = run_analysis(fp)
            # Verileri sakla — konuşma dışa aktarımı için
            self._data = dict(sig=sig, fs=fs, labels=labels, fl=fl, hl=hl,
                              wav_path=fp, stats=st)
            self._set_stats(st)
            self._plot(sig, fs, energy, zcr, labels, fl, hl)
            self.btn_save.config(state="normal")
            self.btn_export.config(state="normal")
            self.status.config(
                text=f"  ✅ Analiz tamamlandı — {os.path.basename(fp)}  |  "
                     f"V:{st['nv']}  UV:{st['nu']}  S:{st['ns']}")
        except Exception as e:
            messagebox.showerror("Hata", f"Dosya analiz edilemedi:\n\n{e}")
            self.status.config(text=f"  ❌ Hata: {e}")

    # ─── KART GÜNCELLE ───────────────────────────────────────────────────
    def _set_stats(self, s):
        self._cards["dur"].config(text=f"{s['dur']:.2f} s")
        self._cards["fs"].config(text=f"{s['fs']} Hz")
        self._cards["nfr"].config(text=str(s['n_fr']))
        self._cards["pv"].config(text=f"%{s['pv']:.1f}")
        self._cards["pu"].config(text=f"%{s['pu']:.1f}")
        self._cards["ps"].config(text=f"%{s['ps']:.1f}")

    # ─── GRAFİK ÇİZ ─────────────────────────────────────────────────────
    def _plot(self, sig, fs, energy, zcr, labels, fl, hl):
        for w in self.plot_area.winfo_children(): w.destroy()

        nf = len(energy)
        t_sig = np.arange(len(sig)) / fs
        t_fr  = (np.arange(nf) * hl + fl / 2) / fs

        # Dark-theme matplotlib ayarları
        rc = {"figure.facecolor": C["plot_bg"], "axes.facecolor": C["plot_bg"],
              "axes.edgecolor": C["grid"], "axes.labelcolor": C["text"],
              "text.color": C["text"], "xtick.color": C["dim"],
              "ytick.color": C["dim"], "grid.color": C["grid"], "grid.alpha": 0.3}
        plt.rcParams.update(rc)

        fig = Figure(figsize=(14, 8), dpi=100, facecolor=C["plot_bg"])
        fig.subplots_adjust(hspace=0.38, top=0.94, bottom=0.07, left=0.06, right=0.94)
        axes = fig.subplots(3, 1, sharex=True)

        # ── Panel 1: Orijinal Sinyal ──
        axes[0].plot(t_sig, sig, color=C["blue"], lw=0.45, alpha=0.9)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Orijinal Sinyal (Normalize)", fontsize=11,
                          fontweight="bold", color=C["text"], pad=8)
        axes[0].set_ylim(-1.08, 1.08); axes[0].grid(True)

        # ── Panel 2: Enerji & ZCR ──
        ax2 = axes[1]; ax2z = ax2.twinx()
        ax2.fill_between(t_fr, energy, alpha=0.30, color=C["red"])
        ax2.plot(t_fr, energy, color=C["red"], lw=1.0)
        ax2z.plot(t_fr, zcr, color=C["orange"], lw=1.0, ls="--", alpha=0.9)
        ax2.set_ylabel("Karesel Enerji", color=C["red"])
        ax2z.set_ylabel("ZCR", color=C["orange"])
        axes[1].set_title("Pencere Bazlı Enerji ve Zero Crossing Rate",
                          fontsize=11, fontweight="bold", color=C["text"], pad=8)
        axes[1].grid(True)
        ax2.legend(handles=[
            Line2D([0],[0], color=C["red"], lw=2, label="Karesel Enerji"),
            Line2D([0],[0], color=C["orange"], lw=2, ls="--", label="ZCR"),
        ], loc="upper right", fontsize=8, facecolor=C["card"],
           edgecolor=C["grid"], labelcolor=C["text"])

        # ── Panel 3: Voiced / Unvoiced Maske ──
        ax3 = axes[2]
        ax3.plot(t_sig, sig, color=C["dim"], lw=0.35, alpha=0.5)
        ax3.set_ylabel("Amplitude"); ax3.set_xlabel("Zaman (s)")
        ax3.set_title("Voiced / Unvoiced / Sessizlik Segmentasyonu",
                       fontsize=11, fontweight="bold", color=C["text"], pad=8)
        ax3.set_ylim(-1.08, 1.08); ax3.grid(True)

        # Bitişik aynı etiketleri birleştir → hızlı çizim
        i = 0
        while i < nf:
            lb = labels[i]; j = i
            while j < nf and labels[j] == lb: j += 1
            ts = i * hl / fs; te = ((j-1)*hl + fl) / fs
            if lb == 2:   ax3.axvspan(ts, te, color=C["green"],  alpha=0.28)
            elif lb == 1: ax3.axvspan(ts, te, color=C["yellow"], alpha=0.32)
            i = j

        ax3.legend(handles=[
            Patch(facecolor=C["green"],  alpha=0.5, label="Voiced"),
            Patch(facecolor=C["yellow"], alpha=0.5, label="Unvoiced"),
            Patch(facecolor=C["plot_bg"], edgecolor=C["grid"], label="Sessizlik"),
        ], loc="upper right", fontsize=8, facecolor=C["card"],
           edgecolor=C["grid"], labelcolor=C["text"])

        # Canvas'ı Tkinter'a göm
        canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = tk.Frame(self.plot_area, bg=C["surface"])
        tb_frame.pack(fill="x")
        tb = NavigationToolbar2Tk(canvas, tb_frame)
        tb.configure(background=C["surface"]); tb.update()

        self._fig = fig

    # ─── PNG KAYDET ──────────────────────────────────────────────────────
    def _save(self):
        if not hasattr(self, "_fig"): return
        fp = filedialog.asksaveasfilename(
            title="Grafiği Kaydet", defaultextension=".png",
            filetypes=[("PNG","*.png"),("SVG","*.svg"),("PDF","*.pdf")],
            initialfile="speech_analysis_result.png")
        if fp:
            self._fig.savefig(fp, dpi=200, facecolor=C["plot_bg"])
            self.status.config(text=f"  💾 Grafik kaydedildi: {fp}")

    # ─── KONUŞMAYI WAV OLARAK DIŞA AKTAR ────────────────────────────────
    def _export_speech(self):
        if not hasattr(self, "_data"): return
        d = self._data

        # Varsayılan dosya adı: orijinal_speech_only.wav
        base = os.path.splitext(os.path.basename(d["wav_path"]))[0]
        default_name = f"{base}_speech_only.wav"

        fp = filedialog.asksaveasfilename(
            title="Konuşma Bölgelerini WAV Olarak Kaydet",
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")],
            initialfile=default_name)
        if not fp: return

        try:
            info = save_speech_wav(
                wav_path=d["wav_path"], output_path=fp,
                signal=d["sig"], fs=d["fs"],
                labels=d["labels"], frame_len=d["fl"], hop_len=d["hl"])

            msg = (f"Sessizlik %{info['pct_removed']:.1f} oranında temizlendi.\n\n"
                   f"Orijinal süre : {info['orig_dur']:.2f} s\n"
                   f"Yeni dosya    : {info['new_dur']:.2f} s\n"
                   f"Kaldırılan    : {info['removed_dur']:.2f} s\n\n"
                   f"Kaydedildi: {fp}")
            messagebox.showinfo("Konuşma Kaydedildi", msg)
            self.status.config(
                text=f"  🔊 Konuşma kaydedildi: {os.path.basename(fp)}  |  "
                     f"Sessizlik %{info['pct_removed']:.1f} temizlendi  |  "
                     f"{info['new_dur']:.2f}s")
        except Exception as e:
            messagebox.showerror("Hata", f"Konuşma kaydedilemedi:\n\n{e}")
            self.status.config(text=f"  ❌ Hata: {e}")

    def run(self):
        self.root.mainloop()


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    App().run()
