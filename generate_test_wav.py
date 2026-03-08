"""
Sentetik Test WAV Dosyasi Olusturucu
=====================================
Bu script, speech_analysis.py'yi test etmek icin icinde
sessizlik, voiced ve unvoiced bolgeler bulunan sentetik
bir .wav dosyasi olusturur.

Yapi (toplam ~3 saniye):
  0.0 - 0.5s : Sessizlik (hafif arka plan gurultusu)
  0.5 - 1.0s : Voiced  — 150 Hz temel frekansli ses teli simulasyonu (A unlusi)
  1.0 - 1.3s : Sessizlik
  1.3 - 1.7s : Unvoiced — Beyaz gurultu (S/F sesi simulasyonu)
  1.7 - 2.0s : Sessizlik
  2.0 - 2.6s : Voiced  — 120 Hz (O unlusi)
  2.6 - 3.0s : Sessizlik
"""

import numpy as np
import wave
import struct

# --- Parametreler ---
FS = 16000          # Ornekleme hizi (16 kHz — konusma icin standart)
DURATION = 3.0      # Toplam sure (saniye)
OUTPUT = "test_audio.wav"

num_samples = int(FS * DURATION)
t = np.arange(num_samples) / FS   # Zaman ekseni

signal = np.zeros(num_samples, dtype=np.float64)

# --- Hafif arka plan gurultusu (tum sinyale) ---
noise_floor = 0.005
signal += np.random.randn(num_samples) * noise_floor

# --- Bolge 1: Voiced (0.5s - 1.0s) — 'A' unlüsü simulasyonu ---
# Ses teli titresimi: temel frekans + harmonikler
start1, end1 = int(0.5 * FS), int(1.0 * FS)
f0 = 150  # Hz — temel frekans
for harmonic in range(1, 5):  # 1., 2., 3., 4. harmonik
    amplitude = 0.6 / harmonic  # Harmonikler azalan genlikle
    signal[start1:end1] += amplitude * np.sin(
        2 * np.pi * f0 * harmonic * t[start1:end1]
    )

# --- Bolge 2: Unvoiced (1.3s - 1.7s) — 'S/F' sesi simulasyonu ---
# Beyaz gurultu — yuksek ZCR, dusuk enerji
start2, end2 = int(1.3 * FS), int(1.7 * FS)
signal[start2:end2] += np.random.randn(end2 - start2) * 0.15

# --- Bolge 3: Voiced (2.0s - 2.6s) — 'O' unlüsü simulasyonu ---
start3, end3 = int(2.0 * FS), int(2.6 * FS)
f0_2 = 120  # Hz — biraz daha dusuk pitch
for harmonic in range(1, 4):
    amplitude = 0.5 / harmonic
    signal[start3:end3] += amplitude * np.sin(
        2 * np.pi * f0_2 * harmonic * t[start3:end3]
    )

# --- Normalize [-1, 1] ---
max_val = np.max(np.abs(signal))
if max_val > 0:
    signal = signal / max_val

# --- 16-bit PCM WAV olarak kaydet ---
signal_int16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)

with wave.open(OUTPUT, 'w') as wf:
    wf.setnchannels(1)        # Mono
    wf.setsampwidth(2)        # 16-bit
    wf.setframerate(FS)       # 16 kHz
    wf.writeframes(signal_int16.tobytes())

print(f"[OK] Sentetik WAV dosyasi olusturuldu: {OUTPUT}")
print(f"     Ornekleme hizi : {FS} Hz")
print(f"     Sure           : {DURATION} s")
print(f"     Ornekler       : {num_samples}")
print(f"     Yapi:")
print(f"       0.0-0.5s  Sessizlik")
print(f"       0.5-1.0s  Voiced (A — 150 Hz)")
print(f"       1.0-1.3s  Sessizlik")
print(f"       1.3-1.7s  Unvoiced (S/F gurultusu)")
print(f"       1.7-2.0s  Sessizlik")
print(f"       2.0-2.6s  Voiced (O — 120 Hz)")
print(f"       2.6-3.0s  Sessizlik")
